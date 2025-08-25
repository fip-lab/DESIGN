#!/bin/bash
# 目标时间：2025年1月2日凌晨0:30
TARGET_TIME="2025-01-02 00:30:00"

# 将目标时间转换为时间戳
TARGET_TIMESTAMP=$(date -d "$TARGET_TIME" +%s)

# 获取当前时间的时间戳
CURRENT_TIMESTAMP=$(date +%s)

# 计算需要等待的秒数
WAIT_SECONDS=$((TARGET_TIMESTAMP - CURRENT_TIMESTAMP))

# 如果目标时间已经过去，则退出
if [ $WAIT_SECONDS -lt 0 ]; then
    echo "目标时间已经过去，无法执行任务。"
    exit 1
fi

# 等待到目标时间
echo "等待到目标时间：$TARGET_TIME"
sleep $WAIT_SECONDS

nohup python3 inference_dinstruct_llama_0shot.py \
    --model_name "llama3.3-70b-instruct" \
    --api_key "sk-f0e64b37da6d4f64ba4094eaf057df03" \
    --save_file "llama3.3-70b-instruct-0shot.json" \
> llama3.3-70b-instruct-0shot.log 2>&1 &

nohup python3 inference_dinstruct_llama.py \
    --model_name "llama3.3-70b-instruct" \
    --api_key "sk-4db634be81fd4d8abe109338f35358c3" \
    --save_file "llama3.3-70b-instruct-1shot.json" \
> llama3.3-70b-instruct-1shot.log 2>&1 &

