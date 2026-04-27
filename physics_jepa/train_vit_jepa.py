import argparse
from pathlib import Path
from omegaconf import OmegaConf

from .train import Trainer
from .utils.hydra import compose

class JepaViTTrainer(Trainer):
    def __init__(self, cfg):
        super().__init__(cfg)

    def pred_fn(self, batch, model_components, loss_fn):
        encoder, predictor = model_components
        ctx_embed = encoder(batch['context'])
        tgt_embed = encoder(batch['target'])
        pred = predictor(ctx_embed)

        # Compute loss on projected embeddings
        loss_dict = loss_fn(ctx_embed, pred, tgt_embed)

        return pred, loss_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, default=f"{Path(__file__).parent.parent}/configs/train_grayscott.yml")
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--encoder_path", type=str, default=None)
    parser.add_argument("--predictor_path", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    cfg = compose(args.config, args.overrides)
    OmegaConf.set_struct(cfg, False)
    cfg.dry_run = args.dry_run
    # cfg.train.encoder_path = args.encoder_path
    # cfg.train.predictor_path = args.predictor_path
    
    cfg.model.objective = "vit_jepa"

    print(OmegaConf.to_yaml(cfg, resolve=True))

    trainer = JepaViTTrainer(cfg)
    trainer.train()
