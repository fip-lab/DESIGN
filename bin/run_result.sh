#!/bin/bash

dataroot="/disk4/zhuangjt/zhuangjt_disk3/SELECT/data/SK-TOD/divide_0"
eval_dataset=test
# verify and evaluate the output
scorefile=scores.json
outfile=predictions.json

rg_output_file=${dataroot}/${eval_dataset}/${outfile}
rg_output_score_file=${dataroot}/${eval_dataset}/${scorefile}
python -m scripts.scores --dataset ${eval_dataset} --dataroot ${dataroot} --outfile ${rg_output_file} --scorefile ${rg_output_score_file}



dataroot="/disk4/zhuangjt/zhuangjt_disk3/SELECT/data/SK-TOD/divide_1"
eval_dataset=test
# verify and evaluate the output
scorefile=scores.json
outfile=predictions.json

rg_output_file=${dataroot}/${eval_dataset}/${outfile}
rg_output_score_file=${dataroot}/${eval_dataset}/${scorefile}
python -m scripts.scores --dataset ${eval_dataset} --dataroot ${dataroot} --outfile ${rg_output_file} --scorefile ${rg_output_score_file}



dataroot="/disk4/zhuangjt/zhuangjt_disk3/SELECT/data/SK-TOD/divide_2"
eval_dataset=test
# verify and evaluate the output
scorefile=scores.json
outfile=predictions.json

rg_output_file=${dataroot}/${eval_dataset}/${outfile}
rg_output_score_file=${dataroot}/${eval_dataset}/${scorefile}
python -m scripts.scores --dataset ${eval_dataset} --dataroot ${dataroot} --outfile ${rg_output_file} --scorefile ${rg_output_score_file}


dataroot="/disk4/zhuangjt/zhuangjt_disk3/SELECT/data/SK-TOD/divide_3"
eval_dataset=test
# verify and evaluate the output
scorefile=scores.json
outfile=predictions.json

rg_output_file=${dataroot}/${eval_dataset}/${outfile}
rg_output_score_file=${dataroot}/${eval_dataset}/${scorefile}
python -m scripts.scores --dataset ${eval_dataset} --dataroot ${dataroot} --outfile ${rg_output_file} --scorefile ${rg_output_score_file}


dataroot="/disk4/zhuangjt/zhuangjt_disk3/SELECT/data/SK-TOD/divide_4"
eval_dataset=test
# verify and evaluate the output
scorefile=scores.json
outfile=predictions.json

rg_output_file=${dataroot}/${eval_dataset}/${outfile}
rg_output_score_file=${dataroot}/${eval_dataset}/${scorefile}
python -m scripts.scores --dataset ${eval_dataset} --dataroot ${dataroot} --outfile ${rg_output_file} --scorefile ${rg_output_score_file}