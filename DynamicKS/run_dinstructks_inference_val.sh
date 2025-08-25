#!/bin/bash

nohup python3 DInstructKS_inference.py \
    --device "cuda:0" \
    --dataset_split "val" \
    --positive_sample_size 2\
    --negative_sample_size 0\
    --model_name 'QuaSiKS_entity_2pos' \
> entity_2pos_val.log 2>&1 &
first_process=$!

wait ${first_process}
nohup python3 DInstructKS_inference.py \
    --device "cuda:0" \
    --dataset_split "val" \
    --positive_sample_size 3\
    --negative_sample_size 0\
    --model_name 'QuaSiKS_entity_3pos' \
> entity_3pos_val.log 2>&1 &
second_process=$!

wait ${second_process}
nohup python3 DInstructKS_inference.py \
    --device "cuda:0" \
    --dataset_split "val" \
    --positive_sample_size 4\
    --negative_sample_size 0\
    --model_name 'QuaSiKS_entity_4pos' \
> entity_4pos_val.log 2>&1 &
third_process=$!

wait ${third_process}
nohup python3 DInstructKS_inference.py \
    --device "cuda:0" \
    --dataset_split "val" \
    --positive_sample_size 5\
    --negative_sample_size 0\
    --model_name 'QuaSiKS_entity_5pos' \
> entity_5pos_val.log 2>&1 &
forth_process=$!

wait ${forth_process}
nohup python3 DInstructKS_inference.py \
    --device "cuda:0" \
    --dataset_split "val" \
    --positive_sample_size 6\
    --negative_sample_size 0\
    --model_name 'QuaSiKS_entity_6pos' \
> entity_6pos_val.log 2>&1 &