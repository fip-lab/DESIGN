#!/bin/bash




nohup python3 EIScore.py --response_file '/home/zhuangjt/zhuangjt_disk3/SELECT/result/test/Bart-base_withABSA_withSelect_2.jsonl' --dialogue_file '/home/zhuangjt/zhuangjt_disk3/SELECT/data/ReDial/ReDial_Postprocess/test_data.jsonl' --out_file '/home/zhuangjt/zhuangjt_disk3/SELECT/score/test/Bart-base_withABSA_withSelect_EIScore.jsonl' --device "cuda:0" > Bart-base_withABSA_withSelect_EIScore.log 2>&1 &

nohup python3 EIScore.py --response_file '/home/zhuangjt/zhuangjt_disk3/SELECT/result/test/Bart-base_withABSA_2.jsonl' --dialogue_file '/home/zhuangjt/zhuangjt_disk3/SELECT/data/ReDial/ReDial_Postprocess/test_data.jsonl'  --out_file '/home/zhuangjt/zhuangjt_disk3/SELECT/score/test/Bart-base_withABSA_EIScore.jsonl' --device "cuda:1" > Bart-base_withABSA_EIScore.log 2>&1 &
first_process=$!

wait ${first_process}


nohup python3 EIScore.py --response_file '/home/zhuangjt/zhuangjt_disk3/SELECT/result/test/Bart-base_withSelect_2.jsonl' --dialogue_file '/home/zhuangjt/zhuangjt_disk3/SELECT/data/ReDial/ReDial_Postprocess/test_data.jsonl'  --out_file '/home/zhuangjt/zhuangjt_disk3/SELECT/score/test/Bart-base_withSelect_EIScore.jsonl' --device "cuda:0" > Bart-base_withSelect_EIScore.log 2>&1 &

nohup python3 EIScore.py --response_file '/home/zhuangjt/zhuangjt_disk3/SELECT/result/test/Bart-base_2.jsonl' --dialogue_file '/home/zhuangjt/zhuangjt_disk3/SELECT/data/ReDial/ReDial_Postprocess/test_data.jsonl'  --out_file '/home/zhuangjt/zhuangjt_disk3/SELECT/score/test/Bart-base_EIScore.jsonl' --device "cuda:1" > Bart-base_EIScore.log 2>&1 &

second_process=$!

wait ${second_process}


nohup python3 EIScore.py --response_file '/home/zhuangjt/zhuangjt_disk3/SELECT/result/val/Bart-base_withABSA_withSelect_2.jsonl' --dialogue_file '/home/zhuangjt/zhuangjt_disk3/SELECT/data/ReDial/ReDial_Postprocess/val_data.jsonl' --out_file '/home/zhuangjt/zhuangjt_disk3/SELECT/score/val/Bart-base_withABSA_withSelect_EIScore.jsonl' --device "cuda:0" > Bart-base_withABSA_withSelect_EIScore.log 2>&1 &

nohup python3 EIScore.py --response_file '/home/zhuangjt/zhuangjt_disk3/SELECT/result/val/Bart-base_withABSA_2.jsonl' --dialogue_file '/home/zhuangjt/zhuangjt_disk3/SELECT/data/ReDial/ReDial_Postprocess/val_data.jsonl'  --out_file '/home/zhuangjt/zhuangjt_disk3/SELECT/score/val/Bart-base_withABSA_EIScore.jsonl' --device "cuda:1" > Bart-base_withABSA_EIScore.log 2>&1 &

third_process=$!

wait ${third_process}

nohup python3 EIScore.py --response_file '/home/zhuangjt/zhuangjt_disk3/SELECT/result/val/Bart-base_withSelect_2.jsonl' --dialogue_file '/home/zhuangjt/zhuangjt_disk3/SELECT/data/ReDial/ReDial_Postprocess/val_data.jsonl'  --out_file '/home/zhuangjt/zhuangjt_disk3/SELECT/score/val/Bart-base_withSelect_EIScore.jsonl' --device "cuda:0" > Bart-base_withSelect_EIScore.log 2>&1 &

nohup python3 EIScore.py --response_file '/home/zhuangjt/zhuangjt_disk3/SELECT/result/val/Bart-base_2.jsonl' --dialogue_file '/home/zhuangjt/zhuangjt_disk3/SELECT/data/ReDial/ReDial_Postprocess/val_data.jsonl'  --out_file '/home/zhuangjt/zhuangjt_disk3/SELECT/score/val/Bart-base_EIScore.jsonl' --device "cuda:1" > Bart-base_EIScore.log 2>&1 &