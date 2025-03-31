#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python ./inference.py \
    --model_path "./checkpoints/EventGPT-7b" \
    --event_frame "./samples/sample1.npy" \
    --query "Describe in detail what happened in the scene." \
    --temperature "0.4" \
    --top_p "1" \