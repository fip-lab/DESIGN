#!/bin/bash

nohup python3 DInstructKS_inference_reviewer.py \
    --device "cuda:0" \
    --dataset_split "test" \
    --positive_sample_size 2\
    --negative_sample_size 0\
    --model_name 'QuaSiKS_reviewer_2pos' \
> reviewer_2pos_test.log 2>&1 &
first_process=$!

wait ${first_process}
nohup python3 DInstructKS_inference_reviewer.py \
    --device "cuda:0" \
    --dataset_split "test" \
    --positive_sample_size 3\
    --negative_sample_size 0\
    --model_name 'QuaSiKS_reviewer_3pos' \
> reviewer_3pos_test.log 2>&1 &
second_process=$!

wait ${second_process}
nohup python3 DInstructKS_inference_reviewer.py \
    --device "cuda:0" \
    --dataset_split "test" \
    --positive_sample_size 4\
    --negative_sample_size 0\
    --model_name 'QuaSiKS_reviewer_4pos' \
> reviewer_4pos_test.log 2>&1 &
third_process=$!

wait ${third_process}
nohup python3 DInstructKS_inference_reviewer.py \
    --device "cuda:0" \
    --dataset_split "test" \
    --positive_sample_size 5\
    --negative_sample_size 0\
    --model_name 'QuaSiKS_reviewer_5pos' \
> reviewer_5pos_test.log 2>&1 &
forth_process=$!

wait ${forth_process}
nohup python3 DInstructKS_inference_reviewer.py \
    --device "cuda:0" \
    --dataset_split "test" \
    --positive_sample_size 6\
    --negative_sample_size 0\
    --model_name 'QuaSiKS_reviewer_6pos' \
> reviewer_6pos_test.log 2>&1 &


