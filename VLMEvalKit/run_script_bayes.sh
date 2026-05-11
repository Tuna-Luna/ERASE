#!/bin/bash
GPU_ID=3

POLICY="bayes"
OUT_DIR="./result/bayes/"
ITER="100"
ALPHA="0.65"
INDICES="../models/bayes_search/indices.xlsx"

# Qwen2.5-VL-7B-Instruct
CUDA_VISIBLE_DEVICES=$GPU_ID python run_bayes.py \
    --data $DATASETS \
    --model "Qwen2.5-VL-7B-Instruct" \
    --policy $POLICY \
    --bayes-iter $ITER \
    --alpha $ALPHA \
    --file-path $INDICES \
    --work-dir $OUT_DIR
sleep 5