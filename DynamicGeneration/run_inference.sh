#!/bin/bash
dataset_type="test"
knowledge_organization_mode="orginal"
with_absa="False"
checkpoint="./checkpoint3/orginal_False_1205/checkpoint-18460"
ks_labels_path="pred/${dataset_type}/DebertaKS_QuaSiKS_reviewer_all_without_metadata.json"
output_path="/home/zhuangjt/zhuangjt_disk3/SK-TOD/pred/${dataset_type}/flan_generate/flan_generation_${knowledge_organization_mode}_${with_absa}_cp18460.json"
nohup python3 inference.py \
    --knowledge_organization_mode ${knowledge_organization_mode} \
    --with_absa ${with_absa} \
    --dataset_type ${dataset_type} \
    --ks_labels_path ${ks_labels_path} \
    --output_path ${output_path} \
    --checkpoint ${checkpoint} \
> flan_generation_orginal_${knowledge_organization_mode}_${with_absa}_${dataset_type}_cp9230.log 2>&1 &



#!/bin/bash
dataset_type="val"
knowledge_organization_mode="orginal"
with_absa="False"
checkpoint="./checkpoint3/orginal_False_1205/checkpoint-18460"
ks_labels_path="pred/${dataset_type}/DebertaKS_QuaSiKS_reviewer_all_without_metadata.json"
output_path="/home/zhuangjt/zhuangjt_disk3/SK-TOD/pred/${dataset_type}/flan_generate/flan_generation_${knowledge_organization_mode}_${with_absa}_cp9230.json"
nohup python3 inference.py \
    --knowledge_organization_mode ${knowledge_organization_mode} \
    --with_absa ${with_absa} \
    --dataset_type ${dataset_type} \
    --ks_labels_path ${ks_labels_path} \
    --output_path ${output_path} \
    --checkpoint ${checkpoint} \
> flan_generation_orginal_${knowledge_organization_mode}_${with_absa}_${dataset_type}_cp18460.log 2>&1 &


dataset_type="test"
knowledge_organization_mode="orginal"
with_absa="False"
checkpoint="./checkpoint3/orginal_False_1205/checkpoint-3692"
ks_labels_path="pred/${dataset_type}/DebertaKS_QuaSiKS_reviewer_all_without_metadata.json"
output_path="/home/zhuangjt/zhuangjt_disk3/SK-TOD/pred/${dataset_type}/flan_generate/flan_generation_${knowledge_organization_mode}_${with_absa}_cp1846.json"
nohup python3 inference.py \
    --knowledge_organization_mode ${knowledge_organization_mode} \
    --with_absa ${with_absa} \
    --dataset_type ${dataset_type} \
    --ks_labels_path ${ks_labels_path} \
    --output_path ${output_path} \
    --checkpoint ${checkpoint} \
> flan_generation_orginal_${knowledge_organization_mode}_${with_absa}_${dataset_type}_cp3692.log 2>&1 &


#!/bin/bash
dataset_type="val"
knowledge_organization_mode="orginal"
with_absa="False"
checkpoint="./checkpoint3/orginal_False_1205/checkpoint-3692"
ks_labels_path="pred/${dataset_type}/DebertaKS_QuaSiKS_reviewer_all_without_metadata.json"
output_path="/home/zhuangjt/zhuangjt_disk3/SK-TOD/pred/${dataset_type}/flan_generate/flan_generation_${knowledge_organization_mode}_${with_absa}_cp1846json"
nohup python3 inference.py \
    --knowledge_organization_mode ${knowledge_organization_mode} \
    --with_absa ${with_absa} \
    --dataset_type ${dataset_type} \
    --ks_labels_path ${ks_labels_path} \
    --output_path ${output_path} \
    --checkpoint ${checkpoint} \
> flan_generation_orginal_${knowledge_organization_mode}_${with_absa}_${dataset_type}_cp3692.log 2>&1 &
