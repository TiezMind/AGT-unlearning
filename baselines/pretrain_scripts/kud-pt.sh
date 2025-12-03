#!/bin/bash
master_port=18765
model_family=kud-gemma-2-2b-it # kud-gemma-2-2b-it、kud-llama2-7b
lr=3e-4
data_path="../../dataset/KnowUnDo/privacy/full.json"
save_dir="../../paper_models/kud-gemma-2-2b-it_lora_privacy"
num_epochs=10
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node=7 --master_port=$master_port ../pretrain.py --config-name=finetune_lora.yaml batch_size=8 gradient_accumulation_steps=8 model_family=${model_family} lr=${lr} num_epochs=${num_epochs} data_path=${data_path} save_dir=${save_dir}
