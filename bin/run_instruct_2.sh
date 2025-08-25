#!/bin/bash

# 要监控的进程 ID
monitor_pid=1636852

# 检查进程是否存在
if ps -p $monitor_pid > /dev/null; then
    echo "Process $monitor_pid is currently running. Waiting for it to complete..."
    # 持续检查指定的进程是否还在运行
    while ps -p $monitor_pid > /dev/null; do
        sleep 600
    done
    echo "Process $monitor_pid has completed. Starting new tasks."
else
    echo "Process $monitor_pid is not running. Starting new tasks immediately."
fi


positive_sample_size=5
nohup python3 main_DInstructKS.py \
    --positive_sample_size ${positive_sample_size} \
> QuaSiKS_${positive_sample_size}pos_entity.log 2>&1 &
four_second=$!

wait ${four_second}
positive_sample_size=6
nohup python3 main_DInstructKS.py \
    --positive_sample_size ${positive_sample_size} \
> QuaSiKS_${positive_sample_size}pos_entity.log 2>&1 &
