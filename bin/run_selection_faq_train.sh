#!/bin/bash

# training and validation for knowledge selection
model_name=PLM/deberta-v3-large
model_name_exp=deberta-v3-large
cuda_id=1

CUDA_VISIBLE_DEVICES=${cuda_id} nohup python3 baseline.py \
        --task selection \
        --dataroot data \
        --model_name_or_path ${model_name} \
        --negative_sample_method "oracle" \
        --knowledge_file knowledge_with_absa.json \
        --selection_knowledge_type "faq" \
        --params_file baseline/configs/selection/params.json \
        --exp_name ks-${model_name_exp}-oracle-faq\
> logs/ks-deberta-v3-large-oracle-faq.log 2>&1 &