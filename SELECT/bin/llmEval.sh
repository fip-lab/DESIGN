#!/bin/bash

nohup python3 llmEval.py --response_file '/home/zhuangjt/zhuangjt_disk3/SELECT/result/test/Bart-base_withABSA_withSelect_2.jsonl'  --out_file '/home/zhuangjt/zhuangjt_disk3/SELECT/score/test/Bart-base_withABSA_withSelect_llmEval.jsonl' > Bart-base_withABSA_withSelect_llmEval.log 2>&1 &

nohup python3 llmEval.py --response_file '/home/zhuangjt/zhuangjt_disk3/SELECT/result/test/Bart-base_withABSA_2.jsonl'  --out_file '/home/zhuangjt/zhuangjt_disk3/SELECT/score/test/Bart-base_withABSA_llmEval.jsonl' > Bart-base_withABSA_llmEval.log 2>&1 &

nohup python3 llmEval.py --response_file '/home/zhuangjt/zhuangjt_disk3/SELECT/result/test/Bart-base_withSelect_2.jsonl'  --out_file '/home/zhuangjt/zhuangjt_disk3/SELECT/score/test/Bart-base_withSelect_llmEval.jsonl' > Bart-base_withSelect_llmEval.log 2>&1 &

nohup python3 llmEval.py --response_file '/home/zhuangjt/zhuangjt_disk3/SELECT/result/test/Bart-base_2.jsonl'  --out_file '/home/zhuangjt/zhuangjt_disk3/SELECT/score/test/Bart-base_llmEval.jsonl' > Bart-base_llmEval.log 2>&1 &