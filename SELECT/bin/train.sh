#!/bin/bash

nohup python3 train.py --model_name 'bart-large' --withABSA --withSelect --modelSavePath '/home/zhuangjt/zhuangjt_disk3/SELECT/checkpoinnt/Bart-large_withABSA_withSelect' > Bart-large_withABSA_withSelect.log 2>&1 &
first_process=$!

wait ${first_process}
nohup python3 train.py --model_name 'bart-large' --withABSA --modelSavePath '/home/zhuangjt/zhuangjt_disk3/SELECT/checkpoinnt/Bart-large_withABSA' > Bart-large_withABSA.log 2>&1 &
second_process=$!

wait ${second_process}
nohup python3 train.py --model_name 'bart-large' --withSelect --modelSavePath '/home/zhuangjt/zhuangjt_disk3/SELECT/checkpoinnt/Bart-large_withSelect' > Bart-large_withSelect.log 2>&1 &
third_process=$!

wait ${third_process}
nohup python3 train.py --model_name 'bart-large' --modelSavePath '/home/zhuangjt/zhuangjt_disk3/SELECT/checkpoinnt/Bart-large' > Bart-large.log 2>&1 &
fourth_process=$!

wait ${fourth_process}
nohup python3 train.py --model_name 'bart-base' --withABSA --withSelect --modelSavePath '/home/zhuangjt/zhuangjt_disk3/SELECT/checkpoinnt/Bart-base_withABSA_withSelect' > Bart-base_withABSA_withSelect.log 2>&1 &
five_process=$!

wait ${five_process}
nohup python3 train.py --model_name 'bart-base' --withABSA --modelSavePath '/home/zhuangjt/zhuangjt_disk3/SELECT/checkpoinnt/Bart-base_withABSA' > Bart-base_withABSA.log 2>&1 &
seven_process=$!

wait ${seven_process}
nohup python3 train.py --model_name 'bart-base' --withSelect --modelSavePath '/home/zhuangjt/zhuangjt_disk3/SELECT/checkpoinnt/Bart-base_withSelect' > Bart-base_withSelect.log 2>&1 &
eight_process=$!

wait ${eight_process}
nohup python3 train.py --model_name 'bart-base' --modelSavePath '/home/zhuangjt/zhuangjt_disk3/SELECT/checkpoinnt/Bart-base' > Bart-base.log 2>&1 &
