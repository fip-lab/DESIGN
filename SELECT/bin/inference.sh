#!/bin/bash

nohup python3 inference.py --model_name 'Bart-base_withABSA_withSelect/checkpoint-5790' --withABSA --withSelect --resultSavePath 'Bart-base_withABSA_withSelect_1' --device "cuda:0" > Bart-base_withABSA_withSelect_1.log 2>&1 &
nohup python3 inference.py --model_name 'Bart-base_withABSA_withSelect/checkpoint-4632' --withABSA --withSelect --resultSavePath 'Bart-base_withABSA_withSelect_2' --device "cuda:1" > Bart-base_withABSA_withSelect_2.log 2>&1 &

nohup python3 inference.py --model_name 'Bart-base_withABSA/checkpoint-5790' --withABSA --resultSavePath 'Bart-base_withABSA_1' --device "cuda:0" > Bart-base_withABSA_1.log 2>&1 &
nohup python3 inference.py --model_name 'Bart-base_withABSA/checkpoint-4632' --withABSA --resultSavePath 'Bart-base_withABSA_2' --device "cuda:1" > Bart-base_withABSA_2.log 2>&1 &


nohup python3 inference.py --model_name 'Bart-base_withSelect/checkpoint-5790' --withSelect --resultSavePath 'Bart-base_withSelect_1' --device "cuda:0" > Bart-base_withSelect_1.log 2>&1 &
nohup python3 inference.py --model_name 'Bart-base_withSelect/checkpoint-4632' --withSelect --resultSavePath 'Bart-base_withSelect_2' --device "cuda:1" > Bart-base_withSelect_2.log 2>&1 &

nohup python3 inference.py --model_name 'Bart-base/checkpoint-5790' --resultSavePath 'Bart-base_1' --device "cuda:0" > Bart-base_1.log 2>&1 &
nohup python3 inference.py --model_name 'Bart-base/checkpoint-4632' --resultSavePath 'Bart-base_2' --device "cuda:1" > Bart-base_2.log 2>&1 &