#!/bin/bash



nohup python3 train.py --top_k_shot 1 --model_name 'bart-large' --select_algorithm 'SELECT' --data_path "/disk4/zhuangjt/zhuangjt_disk3/SELECT/data/SK-TOD/divide_3" --output_model_path "bart-large-1-shot-divide3" >train_divide3.log 2>&1 &

