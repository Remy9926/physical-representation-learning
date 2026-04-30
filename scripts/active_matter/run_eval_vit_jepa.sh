#!/bin/bash
source "$(dirname "$0")/../env_setup.sh"

# Pass the pretrained checkpoint path as $1 and the pretrained predictor as $2 and head type as $3
python -m physics_jepa.eval \
    --trained_model_path $1 \
    --trained_predictor_path $2 \
    --head_type $3