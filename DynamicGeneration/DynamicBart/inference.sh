#!/bin/bash
dataset="test"



model_path="/disk4/zhuangjt/zhuangjt_disk3/SK-TOD/DynamicGeneration/DynamicBart/results/5divide/bart-large-1-shot-divide0/checkpoint-7899"
log_file="/disk4/zhuangjt/zhuangjt_disk3/SELECT/data/SK-TOD/divide_0/test/logs.json"
label_file="/disk4/zhuangjt/zhuangjt_disk3/SELECT/data/SK-TOD/divide_0/test/labels.json"
out_file="/disk4/zhuangjt/zhuangjt_disk3/SELECT/data/SK-TOD/divide_0/test/predictions.json"
device="cuda:0"
nohup python3 inference.py --top_k_shot 1 --model_path ${model_path} --dataset ${dataset} --log_file ${log_file} --label_ks ${label_file} --out_file ${out_file} --device ${device} --select_algorithm 'SELECT' >inference_divide0.log 2>&1 &



model_path="/disk4/zhuangjt/zhuangjt_disk3/SK-TOD/DynamicGeneration/DynamicBart/results/5divide/bart-large-1-shot-divide1/checkpoint-9669"
log_file="/disk4/zhuangjt/zhuangjt_disk3/SELECT/data/SK-TOD/divide_1/test/logs.json"
label_file="/disk4/zhuangjt/zhuangjt_disk3/SELECT/data/SK-TOD/divide_1/test/labels.json"
out_file="/disk4/zhuangjt/zhuangjt_disk3/SELECT/data/SK-TOD/divide_1/test/predictions.json"
device="cuda:1"
nohup python3 inference.py --top_k_shot 1 --model_path ${model_path} --dataset ${dataset} --log_file ${log_file} --label_ks ${label_file} --out_file ${out_file} --device ${device} --select_algorithm 'SELECT' >inference_divide1.log 2>&1 &



wait

model_path="/disk4/zhuangjt/zhuangjt_disk3/SK-TOD/DynamicGeneration/DynamicBart/results/5divide/bart-large-1-shot-divide2/checkpoint-7884"
log_file="/disk4/zhuangjt/zhuangjt_disk3/SELECT/data/SK-TOD/divide_2/test/logs.json"
label_file="/disk4/zhuangjt/zhuangjt_disk3/SELECT/data/SK-TOD/divide_2/test/labels.json"
out_file="/disk4/zhuangjt/zhuangjt_disk3/SELECT/data/SK-TOD/divide_2/test/predictions.json"
device="cuda:0"
nohup python3 inference.py --top_k_shot 1 --model_path ${model_path} --dataset ${dataset} --log_file ${log_file} --label_ks ${label_file} --out_file ${out_file} --device ${device} --select_algorithm 'SELECT' >inference_divide2.log 2>&1 &



model_path="/disk4/zhuangjt/zhuangjt_disk3/SK-TOD/DynamicGeneration/DynamicBart/results/5divide/bart-large-1-shot-divide3/checkpoint-8800"
log_file="/disk4/zhuangjt/zhuangjt_disk3/SELECT/data/SK-TOD/divide_3/test/logs.json"
label_file="/disk4/zhuangjt/zhuangjt_disk3/SELECT/data/SK-TOD/divide_3/test/labels.json"
out_file="/disk4/zhuangjt/zhuangjt_disk3/SELECT/data/SK-TOD/divide_3/test/predictions.json"
device="cuda:1"
nohup python3 inference.py --top_k_shot 1 --model_path ${model_path} --dataset ${dataset} --log_file ${log_file} --label_ks ${label_file} --out_file ${out_file} --device ${device} --select_algorithm 'SELECT' >inference_divide3.log 2>&1 &



model_path="/disk4/zhuangjt/zhuangjt_disk3/SK-TOD/DynamicGeneration/DynamicBart/results/5divide/bart-large-1-shot-divide4/checkpoint-8792"
log_file="/disk4/zhuangjt/zhuangjt_disk3/SELECT/data/SK-TOD/divide_4/test/logs.json"
label_file="/disk4/zhuangjt/zhuangjt_disk3/SELECT/data/SK-TOD/divide_4/test/labels.json"
out_file="/disk4/zhuangjt/zhuangjt_disk3/SELECT/data/SK-TOD/divide_4/test/predictions.json"
device="cuda:0"
nohup python3 inference.py --top_k_shot 1 --model_path ${model_path} --dataset ${dataset} --log_file ${log_file} --label_ks ${label_file} --out_file ${out_file} --device ${device} --select_algorithm 'SELECT' >inference_divide4.log 2>&1 &