#!/bin/bash
positive_sample_size=6

nohup python3 main_DInstructKS_reviewer.py \
    --positive_sample_size ${positive_sample_size} \
> QuaSiKS_${positive_sample_size}pos_reviewer_nnn.log 2>&1 &
