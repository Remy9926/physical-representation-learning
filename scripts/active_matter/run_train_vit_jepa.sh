#!/bin/bash
source "/scratch/eh3648/physical-representation-learning/scripts/env_setup.sh"

torchrun --nproc_per_node=1 --standalone \
    -m physics_jepa.train_vit_jepa \
    configs/train_activematter_vit.yaml \
    $1
