#!/bin/bash
master_port=18764
model_family=muse-llama2-7b # kud-gemma-2-2b-it、kud-llama2-7b
lr=3e-4
data_path="../../dataset/MUSE/books/raw/retain1.txt"
save_dir="../../paper_models/muse-llama2-7b-lora-retrain"
num_epochs=10
run_name="muse-llama2-7b-lora-retrain"
CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=$master_port ../pretrain.py --config-name=finetune_lora.yaml batch_size=8 gradient_accumulation_steps=8 model_family=${model_family} lr=${lr} num_epochs=${num_epochs} data_path=${data_path} save_dir=${save_dir} run_name=${run_name}
