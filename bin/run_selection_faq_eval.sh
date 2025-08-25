#!/bin/bash
eval_dataset=test
model_name=PLM/deberta-v3-large
model_name_exp=deberta-v3-large
checkpoint=runs/ks-${model_name_exp}-oracle-faq
ks_output_file=pred/${eval_dataset}/ks-${model_name_exp}-faq.json
em_output_file=pred/${eval_dataset}/em/em_deberta-v3-base.json

cuda_id=1

CUDA_VISIBLE_DEVICES=${cuda_id} python3 baseline.py \
        --eval_only \
        --task selection \
        --model_name_or_path ${model_name_exp} \
        --checkpoint ${checkpoint} \
        --dataroot data \
        --selection_knowledge_type "faq" \
        --eval_dataset ${eval_dataset} \
        --labels_file ${em_output_file} \
        --output_file ${ks_output_file} \
        --knowledge_file knowledge_with_absa.json