import unsloth
from unsloth import FastLanguageModel

import requests, json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

import sys
import os
from trl import SFTTrainer, SFTConfig
import wandb
from datasets import load_from_disk

combined_dataset = load_from_disk('/data/AIfusion/Tony2016Edu/Ruilong_Jin/Program/qwen_program/data/cleaned_qwen3_dataset')


os.environ["WANDB_NOTEBOOK_NAME"] = "qwen_lora.py"
wandb.login(key="ba54109fa822a6954163d8b80716919c0d162214")

run = wandb.init(project='Fine-tune-Qwen-32B-4bit on Combined Dataset',)

# 在导入任何库之前设置环境变量
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["DISABLE_UNSLOTH_STATS"] = "1"  # 禁用 Unsloth 统计

# # 阻止所有网络连接
# import socket
# original_socket = socket.socket
# def guarded_socket(*args, **kwargs):
#     raise RuntimeError("网络访问被阻止：强制离线模式")
# socket.socket = guarded_socket

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

max_seq_length = 8192
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/data/AIfusion/Tony2016Edu/Ruilong_Jin/Program/qwen_program/Qwen3-32B-unsloth-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 32,             #Choose any number > 0! Suggested 8,16,32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,    #Best to choose alpha = rank or rank*2
    lora_dropout = 0,   #Supports any, but = 0 is optimized
    bias = "none",       #Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", #True or "unsloth" for very
    random_state = 3407,
    use_rslora = False,     # We support rank stabilized LoRA
    loftq_config = None,    # And LoftQ
)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = combined_dataset,
    eval_dataset = None,    # Can set up evaluation
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,    # Use GA to mimic batch size
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training runs
        max_steps = 30,
        learning_rate = 2e-4,  # Reduce to 2e-5 for long training runs
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "wandb",    # Use this for WandB etc
    ),
)

trainer_stats = trainer.train()

# @title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")