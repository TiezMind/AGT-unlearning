#!/bin/bash
master_port=31513
set -e

forget_data_path="../../dataset/TOFU/forget10.json"
retain_data_path="../../dataset/TOFU/retain90.json"

idonknow_file_path="../../dataset/idontknow.txt"

model_family=tofu-gemma-2-2b-it
model_path="../../paper_models/tofu-gemma-2-2b-it_lora_privacy/"
lr=1e-4
num_epochs=5
ds_config="../config/ds_z0_config.json"
#"ga" "ga_gdr" "ga_klr" 
loss_types=("ga_gdr") #  "npo"  "ga_gdr_sure" "ga_klr_sure" "npo_gdr" "npo_klr" "npo_gdr_sure" "npo_klr_sure"
max_length=512

for loss_type in "${loss_types[@]}"; do
    echo $loss_type
    save_dir="../../memory/${model_family}_${loss_type}_again_${max_length}_${lr}"
    CUDA_VISIBLE_DEVICES=7 torchrun --nproc_per_node=1 --master_port=$master_port ../unlearn.py --config-name=forget_lora.yaml batch_size=1 gradient_accumulation_steps=8 model_family=${model_family} lr=${lr} model_path=${model_path} forget_data_path=${forget_data_path} retain_data_path=${retain_data_path} idonknow_file_path=${idonknow_file_path} loss_type=${loss_type} ds_config=${ds_config} max_length=${max_length} save_dir=${save_dir} num_epochs=${num_epochs}
done