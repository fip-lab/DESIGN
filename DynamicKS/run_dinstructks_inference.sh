#!/bin/bash

nohup python3 DInstructKS_inference_reviewer.py \
    --device "cuda:0" \
    --dataset_split "test" \
    --positive_sample_size 2\
    --negative_sample_size 2\
    --model_name 'QuaSiKS_reviewer_3pos' \
> reviewer_3pos_test.log 2>&1 &

