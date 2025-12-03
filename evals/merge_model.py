from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import argparse
import torch

parser = argparse.ArgumentParser()

parser.add_argument('--base_model_path', type=str, default='', help='')
parser.add_argument('--adapter_path', type=str, )
parser.add_argument("--save_path", type=str,)

args = parser.parse_args()

# base_model = AutoModelForCausalLM.from_pretrained(args.base_model_path)
base_model = AutoModelForCausalLM.from_pretrained(args.base_model_path,dtype=torch.bfloat16)
print("基础模型精度：", base_model.dtype)
model = PeftModel.from_pretrained(base_model, args.adapter_path)
tok = AutoTokenizer.from_pretrained(args.base_model_path)
merged_model = model.merge_and_unload()
print("合并后模型精度：", merged_model.dtype) 

merged_model.save_pretrained(args.save_path)

# merged_model.save_pretrained(args.save_path)
tok.save_pretrained(args.save_path)
print(f"saved in: {args.save_path}")