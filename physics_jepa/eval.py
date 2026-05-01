from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import IterableDataset
from abc import ABC, abstractmethod
import json
from einops import rearrange

import numpy as np
from pathlib import Path
import wandb
from tqdm import tqdm
import h5py
import psutil
import os
import gc
import time
from typing import List, Sequence
import re
from sklearn.metrics import f1_score
from .utils.misc import distprint

from .data import (
    EmbeddingsDataset,
    get_dataset_metadata,
    get_train_dataloader,
    get_val_dataloader,
    get_test_dataloader
)
from .model import get_model_and_loss_cnn, get_autoencoder, get_model_and_loss_vit
from .utils.model_utils import RegressionHead, RegressionMLP, KNNHead
import argparse
from pathlib import Path
from omegaconf import OmegaConf

from .finetuner import JepaFinetuner, VideoMAEFinetuner, JepaViTFinetuner
from .utils.hydra import compose 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, default=f"{Path(__file__).parent.parent}/configs/train_grayscott.yml")
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--trained_model_path", type=str, default=None)
    parser.add_argument("--trained_predictor_path", type=str, default=None)
    parser.add_argument("--head_type", type=str, default="linear")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--k", type=int, default=7)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    cfg = compose(args.config, args.overrides)
    OmegaConf.set_struct(cfg, False)
    cfg.dry_run = args.dry_run
    cfg.seed = args.seed
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(OmegaConf.to_yaml(cfg, resolve=True))

    trained_model_path = Path(args.trained_model_path)
    trained_predictor_path = Path(args.trained_predictor_path)
    loss_fn = nn.MSELoss()
    label_name = "physical_params"

    if not trained_model_path.exists():
        print("Trained model path does not exist!\nExiting...")
        exit(1)
    if not trained_predictor_path.exists():
        print("Trained predictor path does not exist!\nExiting...")
        exit(1)

    encoder, _, _ = get_model_and_loss_vit(
        None,
        encoder_n_layers=cfg.model.n_layers,
        encoder_n_heads=cfg.model.n_heads,
        predictor_n_layers=cfg.predictor.n_layers,
        predictor_n_heads=cfg.predictor.n_heads,
        encoder_embed_dim=cfg.model.embed_dim,
        predictor_embed_dim=cfg.predictor.embed_dim,
    )
    
    print(f"loading state dict from {trained_model_path}", flush=True)
    state_dict = torch.load(trained_model_path)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    encoder.load_state_dict(state_dict)
    encoder = encoder.to(device)
    encoder.eval()
    cfg.ft.head_type = args.head_type

    wandb.init(project="physics-jepa",
                name="eval model",
                config=OmegaConf.to_container(cfg))

    if cfg.ft.head_type == "linear":
        print("Setting up linear regression probe")
        head = RegressionHead(
            in_dim=cfg.model.embed_dim,
            out_dim=2,
            flatten_first=True,
        )
        state_dict = torch.load(trained_predictor_path, weights_only=False)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        head.load_state_dict(state_dict)
        head = head.to(device)
        head.eval()
    elif cfg.ft.head_type == "knn":
        print("Setting up KNN probe")
        print(f"loading embeddings from {trained_predictor_path}", flush=True)

        # Add retry mechanism for file opening to handle temporary locking issues
        max_retries = 5
        retry_delay = 2  # seconds

        for attempt in range(max_retries):
            try:
                # Load from HDF5 files using memory mapping for efficient memory usage
                # Data is loaded on-demand when accessed, not all at once
                train_file = h5py.File(trained_predictor_path, "r", locking=False)

                # Use memory mapping - don't load data into memory, just get references
                embeddings = train_file["embeddings"][:]
                all_labels = train_file["labels"][:]

                # Store file handles so they can be closed later
                break  # Success, exit retry loop

            except (BlockingIOError, OSError) as e:
                print(
                    f"Attempt {attempt + 1}/{max_retries} failed to open HDF5 files: {e}",
                    flush=True,
                )
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...", flush=True)
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise e

        head = KNNHead(
            embeddings,
            all_labels,
            device,
            k=args.k,
        )

        train_file.close()
    else:
        print("Invalid probe type specified!\nExiting...")
        exit(1)

    test_data_loader = get_test_dataloader(
            cfg.dataset.name,
            cfg.dataset.num_frames,
            cfg.dataset.get("num_examples", None),
            cfg.ft.batch_size,
            resolution=224)
    
    if args.rank == 0:
        distprint(f"Fetched test dataloader with {len(test_data_loader.dataset)} samples", local_rank=args.rank)
    
    loss = 0
    STATS = {
                "means": [-3.0, 9.0],  # alpha, zeta
                "stds": [1.41, 5.16],
        }

    def normalize_labels(x, stats={}):
        means = torch.tensor(stats['means']).to(device)
        stds = torch.tensor(stats['stds']).to(device)
        return (x - means) / stds

    for i, batch in tqdm(enumerate(test_data_loader), desc="Evaluating on test set", total=len(test_data_loader)):
        ctx = batch["context"].to(device)
        labels = batch[label_name].to(device)
        labels = normalize_labels(labels, stats=STATS)
        preds = head(encoder(ctx).mean(dim=1))

        curr_loss = loss_fn(preds, labels)
        loss += curr_loss.item()
    
    loss /= len(test_data_loader.dataset)
    
    wandb.log({"mse_loss": loss})
    print(f"Final MSE Loss on test set: {loss}", flush=True)
    print(f"There are {len(test_data_loader.dataset)} test points")


