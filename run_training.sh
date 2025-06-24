#!/usr/bin/env sh
# Due to the limitation of a single 3060 GPU, we will use a small model and a small batch size.

python train.py \
    --epochs 5 \
    --batch_size 2 \
    --max_length 256 \
    --lr 1e-6 \
    --beta 0.1 \
    --seed 2003 \
    --model_name "HuggingFaceTB/SmolLM-135M-Instruct" \
    --dataset_name "jondurbin/truthy-dpo-v0.1" \
    --wandb_project "dpo"
