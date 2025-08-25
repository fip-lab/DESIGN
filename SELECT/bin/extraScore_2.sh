#!/bin/bash

ref_file="/disk4/zhuangjt/zhuangjt_disk3/SELECT/data/ReDial/ReDial_Postprocess/val_data.jsonl"
hyp_file="/disk4/zhuangjt/zhuangjt_disk3/SELECT/result/val/Bart-base_withSelect_1.jsonl"
out_file="/disk4/zhuangjt/zhuangjt_disk3/SELECT/result/val/Bart-base_withSelect_extrascore.json"
nohup python3 extraScore4RD.py --hyp_file $hyp_file --ref_file $ref_file --out_file $out_file > $out_file.log 2>&1 &
