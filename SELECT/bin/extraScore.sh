#!/bin/bash


dataroot="/disk4/zhuangjt/zhuangjt_disk3/SELECT/data/SK-TOD/divide_0"
eval_dataset=test
# verify and evaluate the output
scorefile=BCGscores.json
outfile=predictions.json
refile=labels.json

ref_file=${dataroot}/${eval_dataset}/${refile}
hyp_file=${dataroot}/${eval_dataset}/${outfile}
out_file=${dataroot}/${eval_dataset}/${scorefile}
nohup python3 extraScore.py --hyp_file $hyp_file --ref_file $ref_file --out_file $out_file > $out_file.log 2>&1 &

wait


dataroot="/disk4/zhuangjt/zhuangjt_disk3/SELECT/data/SK-TOD/divide_1"
eval_dataset=test
# verify and evaluate the output
scorefile=BCGscores.json
outfile=predictions.json
refile=labels.json

ref_file=${dataroot}/${eval_dataset}/${refile}
hyp_file=${dataroot}/${eval_dataset}/${outfile}
out_file=${dataroot}/${eval_dataset}/${scorefile}
nohup python3 extraScore.py --hyp_file $hyp_file --ref_file $ref_file --out_file $out_file > $out_file.log 2>&1 &


wait

dataroot="/disk4/zhuangjt/zhuangjt_disk3/SELECT/data/SK-TOD/divide_2"
eval_dataset=test
# verify and evaluate the output
scorefile=BCGscores.json
outfile=predictions.json
refile=labels.json

ref_file=${dataroot}/${eval_dataset}/${refile}
hyp_file=${dataroot}/${eval_dataset}/${outfile}
out_file=${dataroot}/${eval_dataset}/${scorefile}
nohup python3 extraScore.py --hyp_file $hyp_file --ref_file $ref_file --out_file $out_file > $out_file.log 2>&1 &


wait

dataroot="/disk4/zhuangjt/zhuangjt_disk3/SELECT/data/SK-TOD/divide_3"
eval_dataset=test
# verify and evaluate the output
scorefile=BCGscores.json
outfile=predictions.json
refile=labels.json

ref_file=${dataroot}/${eval_dataset}/${refile}
hyp_file=${dataroot}/${eval_dataset}/${outfile}
out_file=${dataroot}/${eval_dataset}/${scorefile}
nohup python3 extraScore.py --hyp_file $hyp_file --ref_file $ref_file --out_file $out_file > $out_file.log 2>&1 &


wait

dataroot="/disk4/zhuangjt/zhuangjt_disk3/SELECT/data/SK-TOD/divide_4"
eval_dataset=test
# verify and evaluate the output
scorefile=BCGscores.json
outfile=predictions.json
refile=labels.json

ref_file=${dataroot}/${eval_dataset}/${refile}
hyp_file=${dataroot}/${eval_dataset}/${outfile}
out_file=${dataroot}/${eval_dataset}/${scorefile}
nohup python3 extraScore.py --hyp_file $hyp_file --ref_file $ref_file --out_file $out_file > $out_file.log 2>&1 &
