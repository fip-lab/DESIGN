#!/bin/bash


dataroot="/disk4/zhuangjt/zhuangjt_disk3/SELECT/data/SK-TOD/divide_4"
eval_dataset=test
# verify and evaluate the output
scorefile=EIscores.json
outfile=predictions.json

rg_output_file=${dataroot}/${eval_dataset}/${outfile}
rg_output_score_file=${dataroot}/${eval_dataset}/${scorefile}
nohup python3 inference_nli_score_2.py \
    --label_ks ${rg_output_file} \
    --out_file ${rg_output_score_file} \
    --device "cuda:1" \
> divide_4_EI.log 2>&1 &