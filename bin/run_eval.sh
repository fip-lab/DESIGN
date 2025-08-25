#!/bin/bash
eval_dataset=val
ks_output_file=KS-val.json
model_name=facebook/bart-base
model_name_exp=bart-base
checkpoint=runs/rg-review-${model_name_exp}-oracle-baseline
rg_output_file=RG-val.json
cuda_id=0
CUDA_VISIBLE_DEVICES=${cuda_id} python3 baseline.py \
        --task generation \
        --generate runs/rg-review-${model_name_exp}-baseline \
        --generation_params_file baseline/configs/generation/generation_params.json \
        --eval_dataset ${eval_dataset} \
        --dataroot data \
        --labels_file ${ks_output_file} \
        --knowledge_file knowledge.json \
        --output_file ${rg_output_file}

eval_dataset=test
ks_output_file=KS-test.json
model_name=facebook/bart-base
model_name_exp=bart-base
checkpoint=runs/rg-review-${model_name_exp}-oracle-baseline
rg_output_file=RG-test.json
cuda_id=0
CUDA_VISIBLE_DEVICES=${cuda_id} python3 baseline.py \
        --task generation \
        --generate runs/rg-review-${model_name_exp}-baseline \
        --generation_params_file baseline/configs/generation/generation_params.json \
        --eval_dataset ${eval_dataset} \
        --dataroot data \
        --labels_file ${ks_output_file} \
        --knowledge_file knowledge.json \
        --output_file ${rg_output_file}