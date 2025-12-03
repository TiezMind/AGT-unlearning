#!/bin/bash
master_port=31514
set -e

CUDA_ID=0
################################模型训练##################################
# forget_data_path="/data1/relearn/dataset/WMDP/cyber-forget-corpus.jsonl"
# retain_data_path="/data1/relearn/dataset/WMDP/cyber-retain-corpus.jsonl"

# idonknow_file_path="../../dataset/idontknow.txt"

# model_family=wmdp-zephyr-7b-beta
# model_path="/data1/model/zephyr-7b-beta"
# lr=1e-4
# num_epochs=3
# ds_config="../config/ds_z0_config.json"
# # "ga_gdr" "ga_klr" "npo_gdr" "npo_klr" "npo"
# loss_types=("sim_npo_gdr")
# max_length=512
# ema_decay=0.9
# adv_update_threshold=2
# adaptive_threshold_enabled=True
# adv_epsilon=1e-2
# adv_steps=4
# adv_alpha=5e-3
# perturb_layer=4

# for loss_type in "${loss_types[@]}"; do
#     echo $loss_type
#     save_dir="../../memory-wmdp/${model_family}_${loss_type}_${max_length}_${lr}"
#     run_name="${model_family}_${loss_type}_${max_length}_${lr}"
#     CUDA_VISIBLE_DEVICES=${CUDA_ID} torchrun --nproc_per_node=1 --master_port=$master_port ../unlearn.py --config-name=forget_lora.yaml run_name=${run_name} batch_size=1 gradient_accumulation_steps=8 model_family=${model_family} lr=${lr} model_path=${model_path} forget_data_path=${forget_data_path} retain_data_path=${retain_data_path} idonknow_file_path=${idonknow_file_path} loss_type=${loss_type} ds_config=${ds_config} max_length=${max_length} save_dir=${save_dir} num_epochs=${num_epochs} +adv_update_threshold=${adv_update_threshold} +adaptive_threshold_enabled=${adaptive_threshold_enabled} +ema_decay=${ema_decay} +adv_epsilon=${adv_epsilon} +adv_steps=${adv_steps} +adv_alpha=${adv_alpha} +perturb_layer=${perturb_layer}
# done

############################合并模型##################################
cd /data1/relearn/evals
base_model_path="/data1/model/zephyr-7b-beta"

memory_dir="../memory-wmdp"

for adapter_dir in "$memory_dir"/*/; do
  adapter_name=$(basename "$adapter_dir")
  echo "Processing adapter: $adapter_name"

  if [[ "$adapter_name" != "wmdp-zephyr-7b-beta_sim_npo_gdr_512_1e-4" ]]; then
    echo "Skipping $adapter_name as it does not match the target adapter."
    continue
  fi
  
  if [[ "$adapter_name" == *zephyr* ]]  && [[ "$adapter_name" != *-full ]]; then
    for checkpoint_dir in "$adapter_dir"*/; do
      if [[ "$checkpoint_dir" == *checkpoint-375* ]]; then
        checkpoint_name=$(basename "$checkpoint_dir")
        if [[ $checkpoint_name == *full ]]; then
          echo "${checkpoint_name} merged"
          continue
        fi

        save_checkpoint_dir="$adapter_dir/${checkpoint_name}-full"

        if [ -d "$save_checkpoint_dir" ]; then
          echo "Skipping $checkpoint_dir because $save_checkpoint_dir already exists."
          continue
        fi

        CUDA_VISIBLE_DEVICES=${CUDA_ID} python merge_model.py \
          --base_model_path "$base_model_path" \
          --adapter_path "$checkpoint_dir" \
          --save_path "$save_checkpoint_dir"
        echo "Merged $checkpoint_dir into $save_checkpoint_dir"
      fi
    done
  fi
done

# ################################eval##################################
# cd /data1/open-unlearning
# conda activate unlearning
# bash /data1/open-unlearning/scripts/eval_MUSE_news.sh