#!/bin/bash
GPU_ID=0
DATASETS="ChartQA_TEST TextVQA_VAL InfoVQA_VAL DocVQA_VAL"
OUT_DIR="./result"

MODELS_LAYERS="Qwen2.5-VL-7B-Instruct:2,19 InternVL3-8B:2,19 Qwen3-VL-8B-Instruct:2,24 Qwen3-VL-4B-Instruct:2,24 Qwen2.5-VL-3B-Instruct:2,24"

for ENTRY in $MODELS_LAYERS; do
    MODEL=${ENTRY%%:*}
    LAYERS=${ENTRY##*:}
    LAYERS=${LAYERS/,/ }
    for RETAIN in 0.25 0.35 0.45; do
        CUDA_VISIBLE_DEVICES=$GPU_ID python -u run.py \
            --data $DATASETS \
            --model $MODEL \
            --policy erase \
            --retain-ratio $RETAIN \
            --weight 0.2 \
            --edge_tau 0.45 \
            --layer_list "$LAYERS" \
            --work-dir $OUT_DIR/erase_retain_$RETAIN
        sleep 5
    done

done
