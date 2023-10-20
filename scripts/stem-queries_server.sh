#!/bin/bash



input_file=$1
model_path=llama-2-7b-chat/
gpu=$2

CUDA_VISIBLE_DEVICES=$gpu  python3 stem-queries.py \
--input_file $input_file \
--model llama \
--model_path $model_path \
--batch 5