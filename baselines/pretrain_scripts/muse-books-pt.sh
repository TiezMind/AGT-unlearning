#!/bin/bash
CUDA_ID=0
batch_size=4
gradient_accumulation_steps=8
max_length=2048

################################target##################################
# master_port=18765
# model_family=muse-books-llama2-7b
# lr=3e-4
# data_path="../../dataset/MUSE/books/train/full.txt"
# save_dir="../../paper_models/muse-books-${max_length}-llama2-7b-lora16-target-epoch20"
# num_epochs=20
# run_name="muse-books-${max_length}-llama2-7b-lora16-target-epoch20"
# CUDA_VISIBLE_DEVICES=${CUDA_ID} torchrun --nproc_per_node=1 --master_port=$master_port ../pretrain.py --config-name=finetune_lora.yaml batch_size=${batch_size} gradient_accumulation_steps=${gradient_accumulation_steps} max_length=${max_length} model_family=${model_family} lr=${lr} num_epochs=${num_epochs} data_path=${data_path} save_dir=${save_dir} run_name=${run_name}

################################eval##################################
# cd /data1/muse_bench
# CORPUS=books
# model_paths=(
#             "/data1/relearn/paper_models/muse-books-${max_length}-llama2-7b-lora-target-epoch20"
#             )
# for model_path in "${model_paths[@]}"; do
#   name=$(basename ${model_path})
#   echo "Processing: $name"
#   CUDA_VISIBLE_DEVICES=${CUDA_ID} python -m muse_bench.eval \
#     --model_dirs ${model_path} \
#     --names ${name} \
#     --corpus ${CORPUS} \
#     --out_file "/data1/muse_bench/${name}.csv"
# done

################################retrain##################################
cd /data1/relearn/baselines/pretrain_scripts
master_port=18765
model_family=muse-books-llama2-7b 
lr=3e-4
data_path="../../dataset/MUSE/books/raw/retain.txt"
save_dir="../../paper_models/muse-books-${max_length}-llama2-7b-lora-retrain-epoch20"
num_epochs=20
run_name="muse-books-${max_length}-llama2-7b-lora-retrain-epoch20"
CUDA_VISIBLE_DEVICES=${CUDA_ID} torchrun --nproc_per_node=1 --master_port=$master_port ../pretrain.py --config-name=finetune_lora.yaml batch_size=${batch_size} gradient_accumulation_steps=${gradient_accumulation_steps} max_length=${max_length} model_family=${model_family} lr=${lr} num_epochs=${num_epochs} data_path=${data_path} save_dir=${save_dir} run_name=${run_name}

################################eval##################################
# cd /data1/muse_bench
# CORPUS=books
# model_paths=(
#             "/data1/relearn/paper_models/muse-books-${max_length}-llama2-7b-lora-retrain-epoch20"
#             )
# for model_path in "${model_paths[@]}"; do
#   name=$(basename ${model_path})
#   echo "Processing: $name"
#   CUDA_VISIBLE_DEVICES=${CUDA_ID} python -m muse_bench.eval \
#     --model_dirs ${model_path} \
#     --names ${name} \
#     --corpus ${CORPUS} \
#     --out_file "/data1/muse_bench/${name}.csv"
# done


