#!/bin/bash

eval_dataset=test
# verify and evaluate the output


rg_output_file=/home/zhuangjt/zhuangjt_disk3/SK-TOD/DynamicGeneration/DynamicBart/pred/DSTC10/test/bart_large_withoutabsa_2shot.json
rg_output_score_file=/home/zhuangjt/zhuangjt_disk3/SK-TOD/DynamicGeneration/DynamicBart/pred/DSTC10/test/bart_large_withoutabsa_2shot_bmr_score.json
python -m scripts_dstc10.score --dataset ${eval_dataset} --dataroot DSTC10_data --outfile ${rg_output_file} --scorefile ${rg_output_score_file}

rg_output_file=/home/zhuangjt/zhuangjt_disk3/SK-TOD/DynamicGeneration/DynamicBart/pred/DSTC10/test/bart_large_withoutabsa_3shot.json
rg_output_score_file=/home/zhuangjt/zhuangjt_disk3/SK-TOD/DynamicGeneration/DynamicBart/pred/DSTC10/test/bart_large_withoutabsa_3shot_bmr_score.json
python -m scripts_dstc10.score --dataset ${eval_dataset} --dataroot DSTC10_data --outfile ${rg_output_file} --scorefile ${rg_output_score_file}