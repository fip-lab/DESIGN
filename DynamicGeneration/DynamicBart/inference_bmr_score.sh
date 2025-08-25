#!/bin/bash

label_flie="/home/zhuangjt/zhuangjt_disk3/SK-TOD/DSTC10_data/test/labels.json"
hyp_file="/home/zhuangjt/zhuangjt_disk3/SK-TOD/DynamicGeneration/DynamicBart/pred/DSTC10/test/bart_large_withoutabsa_1shot.json"
out_file="/home/zhuangjt/zhuangjt_disk3/SK-TOD/DynamicGeneration/DynamicBart/pred/DSTC10/test/bart_large_withoutabsa_1shot_bmr_score.json"
nohup python3 inference_bmr_score.py \
    --label_file ${label_flie} \
    --hyp_file ${hyp_file} \
    --out_file ${out_file} \
    --device "cuda:0" \
> 1shot_bmr_score_test.log 2>&1 &

