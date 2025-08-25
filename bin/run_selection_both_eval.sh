#!/bin/bash
eval_dataset=Dynamic_Dataset_copy/Resource_RG

model_name=PLM/deberta-v3-large
model_name_exp=deberta-v3-large
checkpoint=runs/ks-deberta-v3-large-both-oracle
ks_output_file=data/${eval_dataset}/DebertaKs.json
em_output_file=data/${eval_dataset}/em.json
cuda_id=0

CUDA_VISIBLE_DEVICES=${cuda_id} python3 baseline.py \
        --eval_only \
        --task selection \
        --model_name_or_path ${model_name_exp} \
        --checkpoint ${checkpoint} \
        --dataroot data \
        --selection_knowledge_type "both" \
        --eval_dataset ${eval_dataset} \
        --labels_file ${em_output_file} \
        --output_file ${ks_output_file} \
        --knowledge_file knowledge_with_absa.json