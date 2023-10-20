#!/bin/bash

input_file=$1

model_path=llama-2-7b-chat/


python3 stem-collection-2.py \
--input_file $input_file \
--model llama \
--model_path $model_path \
--batch 1