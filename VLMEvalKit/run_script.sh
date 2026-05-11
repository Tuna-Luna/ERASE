#!/bin/bash
GPU_ID=3


DATASETS="OCRBench ChartQA_TEST TextVQA_VAL InfoVQA_VAL DocVQA_VAL"
RETAIN=0.25
POLICY="erase"
OUT_DIR="./result/prune_75/"

# Qwen2.5-VL-7B-Instruct
ENTROPY="1.6904912034377264 1.3471717001563197 1.1718597959685415"
STAGE1_RETAIN="0.8267523535511546 0.751410400727526 0.49466854974609853 0.4033912240039451"
CUDA_VISIBLE_DEVICES=$GPU_ID python run.py \
    --data $DATASETS \
    --model "Qwen2.5-VL-7B-Instruct" \
    --vision-token-num $RETAIN \
    --policy $POLICY \
    --entropy-threshold $ENTROPY \
    --retain-ratio $STAGE1_RETAIN \
    --work-dir $OUT_DIR
sleep 5

# Qwen3-VL-8B-Instruct
ENTROPY="1.6102664561128828 0.2269265653651543 0.05622356759744964"
STAGE1_RETAIN="0.8450338604820039 0.7763170296713962 0.7573556106689442 0.19399401785935438"
CUDA_VISIBLE_DEVICES=$GPU_ID python run.py \
    --data $DATASETS \
    --model "Qwen3-VL-8B-Instruct" \
    --vision-token-num $RETAIN \
    --policy $POLICY \
    --entropy-threshold $ENTROPY \
    --retain-ratio $STAGE1_RETAIN \
    --work-dir $OUT_DIR
