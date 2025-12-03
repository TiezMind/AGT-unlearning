#!/bin/bash
master_port=18765
model_family=tofu-llama2-7b # kud-gemma-2-2b-it、kud-llama2-7b
lr=3e-4
data_path="../../dataset/TOFU/full.json"
save_dir="../../paper_models/tofu-llama2-7b-lora-vanilla"
num_epochs=10
run_name="tofu-llama2-7b-lora-vanilla"
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=$master_port ../pretrain.py --config-name=finetune_lora.yaml batch_size=4 gradient_accumulation_steps=8 model_family=${model_family} lr=${lr} num_epochs=${num_epochs} data_path=${data_path} save_dir=${save_dir} run_name=${run_name}
