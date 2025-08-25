#!/bin/bash

#gpt-4o
nohup python3 inference_dinstruct_llm.py \
    --model_name "gpt-4o" \
    --base_url "https://xiaoai.plus/v1" \
    --api_key "sk-VDqhyDZPJ6v4htxawcya3tij1tkcWIHoOxJ3kmN3yqJnuLM1" \
    --save_file "gpt-4o-1shot.json" \
> gpt-4o-1shot.log 2>&1 &

#claude-3-5-sonnet-20241022	
nohup python3 inference_dinstruct_llm.py \
    --model_name "claude-3-5-sonnet-20241022" \
    --base_url "https://xiaoai.plus/v1" \
    --api_key "sk-VDqhyDZPJ6v4htxawcya3tij1tkcWIHoOxJ3kmN3yqJnuLM1" \
    --save_file "claude-3-5-sonnet-20241022-1shot.json" \
> claude-3-5-sonnet-20241022-1shot.log 2>&1 &


#gpt-4-1106-preview
nohup python3 inference_dinstruct_llm.py \
    --model_name "gpt-4-1106-preview" \
    --base_url "https://xiaoai.plus/v1" \
    --api_key "sk-VDqhyDZPJ6v4htxawcya3tij1tkcWIHoOxJ3kmN3yqJnuLM1" \
    --save_file "gpt-4-1106-preview-1shot.json" \
> gpt-4-1106-preview-1shot.log 2>&1 &

wait

#deepseek-chat
nohup python3 inference_dinstruct_llm.py \
    --model_name "deepseek-chat" \
    --base_url "https://api.deepseek.com/v1" \
    --api_key "sk-b687149b808241528029732cf0ec7aec" \
    --save_file "deepseek-V3-1shot.json" \
> deepseek-V3-1shot.log 2>&1 &


#qwq-32b-preview
nohup python3 inference_dinstruct_llm.py \
    --model_name "qwq-32b-preview" \
    --base_url "https://dashscope.aliyuncs.com/compatible-mode/v1" \
    --api_key "sk-4db634be81fd4d8abe109338f35358c3" \
    --save_file "qwq-32b-preview-1shot.json" \
> qwq-32b-preview-1shot.log 2>&1 &

#qwen2.5-72b-instruct
nohup python3 inference_dinstruct_llm.py \
    --model_name "qwen2.5-72b-instruct" \
    --base_url "https://dashscope.aliyuncs.com/compatible-mode/v1" \
    --api_key "sk-4db634be81fd4d8abe109338f35358c3" \
    --save_file "qwen2.5-72b-instruct-1shot.json" \
> qwen2.5-72b-instruct-1shot.log 2>&1 &



#gpt-4o
nohup python3 inference_dinstruct_llm_0shot.py \
    --model_name "gpt-4o" \
    --base_url "https://xiaoai.plus/v1" \
    --api_key "sk-VDqhyDZPJ6v4htxawcya3tij1tkcWIHoOxJ3kmN3yqJnuLM1" \
    --save_file "gpt-4o-0shot.json" \
> gpt-4o-0shot.log 2>&1 &


#qwq-32b-preview
nohup python3 inference_dinstruct_llm_0shot.py \
    --model_name "qwq-32b-preview" \
    --base_url "https://dashscope.aliyuncs.com/compatible-mode/v1" \
    --api_key "sk-4db634be81fd4d8abe109338f35358c3" \
    --save_file "qwq-32b-preview-0shot.json" \
> qwq-32b-preview-0shot.log 2>&1 &



#gpt-4-1106-preview
nohup python3 inference_dinstruct_llm_0shot.py \
    --model_name "gpt-4-1106-preview" \
    --base_url "https://xiaoai.plus/v1" \
    --api_key "sk-VDqhyDZPJ6v4htxawcya3tij1tkcWIHoOxJ3kmN3yqJnuLM1" \
    --save_file "gpt-4-1106-preview-0shot.json" \
> gpt-4-1106-preview-0shot.log 2>&1 &

#deepseek-chat
nohup python3 inference_dinstruct_llm_0shot.py \
    --model_name "deepseek-chat" \
    --base_url "https://api.deepseek.com/v1" \
    --api_key "sk-b687149b808241528029732cf0ec7aec" \
    --save_file "deepseek-V3-0shot.json" \
> deepseek-V3-0shot.log 2>&1 &



#claude-3-5-sonnet-20241022	
nohup python3 inference_dinstruct_llm_0shot.py \
    --model_name "claude-3-5-sonnet-20241022" \
    --base_url "https://xiaoai.plus/v1" \
    --api_key "sk-VDqhyDZPJ6v4htxawcya3tij1tkcWIHoOxJ3kmN3yqJnuLM1" \
    --save_file "claude-3-5-sonnet-20241022-0shot.json" \
> claude-3-5-sonnet-20241022-0shot.log 2>&1 &


#qwen2.5-72b-instruct
nohup python3 inference_dinstruct_llm_0shot.py \
    --model_name "qwen2.5-72b-instruct" \
    --base_url "https://dashscope.aliyuncs.com/compatible-mode/v1" \
    --api_key "sk-4db634be81fd4d8abe109338f35358c3" \
    --save_file "qwen2.5-72b-instruct-0shot.json" \
> qwen2.5-72b-instruct-0shot.log 2>&1 &