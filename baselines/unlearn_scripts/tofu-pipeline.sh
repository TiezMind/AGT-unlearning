#!/bin/bash
master_port=31513
set -e

CUDA_ID=0
################################模型训练##################################
# forget_data_path="../../dataset/TOFU/forget10.json"
# retain_data_path="../../dataset/TOFU/retain90.json"

# idonknow_file_path="../../dataset/idontknow.txt"

# model_family=tofu-llama2-7b
# model_path="../../paper_models/tofu-llama2-7b-lora-vanilla/"
# lr=1e-4
# num_epochs=5
# ds_config="../config/ds_z0_config.json"
# # "ga_gdr" "ga_klr" "npo_gdr" "npo_klr" "npo" 
# loss_types=( "ago_wo_aso" "ago_wo_gbg" "ago_w_hard_aso" "ago_wo_lag") #  
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
#     save_dir="../../memory/${model_family}_${loss_type}_${max_length}_${lr}"
#     run_name="${model_family}_${loss_type}_${max_length}_${lr}"
#     CUDA_VISIBLE_DEVICES=${CUDA_ID} torchrun --nproc_per_node=1 --master_port=$master_port ../unlearn.py --config-name=forget_lora.yaml run_name=${run_name} batch_size=1 gradient_accumulation_steps=8 model_family=${model_family} lr=${lr} model_path=${model_path} forget_data_path=${forget_data_path} retain_data_path=${retain_data_path} idonknow_file_path=${idonknow_file_path} loss_type=${loss_type} ds_config=${ds_config} max_length=${max_length} save_dir=${save_dir} num_epochs=${num_epochs} +adv_update_threshold=${adv_update_threshold} +adaptive_threshold_enabled=${adaptive_threshold_enabled} +ema_decay=${ema_decay} +adv_epsilon=${adv_epsilon} +adv_steps=${adv_steps} +adv_alpha=${adv_alpha} +perturb_layer=${perturb_layer}
# done

############################合并模型##################################
cd /data1/relearn/evals
base_model_path="../paper_models/tofu-llama2-7b-lora-vanilla"

memory_dir="../memory"

for adapter_dir in "$memory_dir"/*/; do
  adapter_name=$(basename "$adapter_dir")
  echo "Processing adapter: $adapter_name"

#   if [[ "$adapter_name" != "tofu-gemma-2-2b-it_AGO_512_1e-4" ]]; then
#     echo "Skipping $adapter_name as it does not match the target adapter."
#     continue
#   fi
  
  if [[ "$adapter_name" == *llama2* ]]  && [[ "$adapter_name" != *-full ]]; then
    for checkpoint_dir in "$adapter_dir"*/; do
      if [[ "$checkpoint_dir" == *checkpoint-250* ]]; then
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

################################模型前向推理##################################
# memory_dir="../memory"

# forget_val_data_path="TOFU/forget10_perturbed.json"
# retain_val_data_path="TOFU/retain_perturbed.json"

# output_file_dir="../tofu-gemma-inf"

# mkdir -p "$output_file_dir"



# for adapter_dir in "$memory_dir"/*; do
#     if [ -d "$adapter_dir" ]; then
#         adapter_name=$(basename "$adapter_dir")

#         if [[ "$adapter_name" == tofu-gemma-2-2b-it* && "$adapter_name" != *-full ]] ; then
#             for checkpoint_dir in "$adapter_dir"/*; do
#                 if [ -d "$checkpoint_dir" ]; then
#                     checkpoint_name=$(basename "$checkpoint_dir")

#                     if [[ "$checkpoint_name" == *-full ]]; then
#                         method="${adapter_name}__${checkpoint_name}"

#                         output_file_forget="$output_file_dir/${method}__forget.json"
#                         output_file_retain="$output_file_dir/${method}__retain.json"

#                         if [ -f "$output_file_forget" ] && [ -f "$output_file_retain" ]; then
#                             echo "Output files for $method already exist. Skipping..."
#                             continue
#                         fi

#                         CUDA_VISIBLE_DEVICES=${CUDA_ID} python generate.py \
#                             --model_path "$checkpoint_dir" \
#                             --forget_val_data_path "$forget_val_data_path" \
#                             --retain_val_data_path "$retain_val_data_path" \
#                             --output_file_forget "$output_file_forget" \
#                             --output_file_retain "$output_file_retain" \
#                             --use_vllm
#                     fi
#                 fi
#             done
#         fi
#     fi
# done


##########################模型评估######################################
# language_model_path="model/gemma-2b" # Path to the HF model before pretraining
# embedding_model_path="model/MiniLM" # Path to the MiniLM model 
# entailment_model_path="model/deberta" # Path to the nli model
# pertubed_forget_data="dataset/TOFU/forget10_perturbed.json"
# pertubed_retain_data="dataset/TOFU/retain_perturbed.json"
# memory_dir="../memory"

# output_dir="../tofu-gemma-eval"

# results_dir="../tofu-gemma-inf"

# if [ ! -d "$output_dir" ]; then
#   mkdir -p "$output_dir"
# fi

# for result_file in "$results_dir"/*_forget.json; do
#     base_name=$(basename "$result_file" "__forget.json")
    
#     forget_path="$results_dir/${base_name}__forget.json"
#     retain_path="$results_dir/${base_name}__retain.json"
    
#     if [ -f "$forget_path" ] && [ -f "$retain_path" ]; then
#         test_model_name="$base_name"
        
#         result_path="$output_dir/${test_model_name}.json"
        
#         if [ -f "$result_path" ]; then
#             echo "Result file for $test_model_name already exists. Skipping..."
#             continue
#         fi

#         CUDA_VISIBLE_DEVICES=${CUDA_ID} python evaluate_tofu.py \
#             --language_model_path "$language_model_path" \
#             --embedding_model_path "$embedding_model_path" \
#             --entailment_model_path "$entailment_model_path" \
#             --test_model_name "$test_model_name" \
#             --forget_path "$forget_path" \
#             --retain_path "$retain_path" \
#             --output_path "$result_path" \
#             --unlearning_model "$model_name" \
#             --pertubed_forget_data "$pertubed_forget_data" \
#             --pertubed_retain_data "$pertubed_retain_data"
#     else
#         echo "Warning: Missing files for $base_name. Skipping..."
#     fi
# done



