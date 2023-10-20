#!/bin/bash



input_file=$1
model_path=llama-2-7b-chat/
gpu=$2

CUDA_VISIBLE_DEVICES=$gpu  python3 entity_recog_queries.py \
--input_file $input_file \
--batch 256