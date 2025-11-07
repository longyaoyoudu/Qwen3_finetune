# Qwen3_finetune
# 环境准备
```python
Unsloth
vLLM
EvalScope
wandb
```
# 一、模型下载

通过魔搭社区下载 Qwen3-unsloth-bnb-4bit模型 
<img width="819" height="406" alt="image" src="https://github.com/user-attachments/assets/c44eca74-5d4c-4b98-a8a5-e807d005dfdc" />

# 二、模型简单调用

使用vLLM对下载好的模型进行简单调用
```bash
vllm serve ./Qwen3-32B-unsloth-bnb-4bit --enable-auto-tool-choice --tool-call-parser hermes --gpu-memory-utilization 0.7
```
<img width="866" height="417" alt="image" src="https://github.com/user-attachments/assets/bd2a8d5a-5377-47d6-8d3e-d240b3987767" />

根据gpu显存大小自定义选择gpu内存利用率，这里以英伟达A800显卡为例
<img width="1226" height="188" alt="image" src="https://github.com/user-attachments/assets/e89bf9ae-af65-4726-bede-9d90d9aa6d4c" />

# 三、模型评估

使用EvalScope对模型进行压力测试
```bash
evalscope perf --url "http://127.0.0.1:8000/v1/chat/completions" --parallel 5 --model ./Qwen3-32B-unsloth-bnb-4bit --number 20 --api openai --dataset openqa --stream
```
<img width="998" height="302" alt="image" src="https://github.com/user-attachments/assets/ec2c177f-25b4-419e-98fa-cef7a9451a24" />

通过运行文件test_eval.py对模型进行压力测试

# 四、微调数据集准备

本项目选择使用OpenMathReasoning和FineTome-100k数据集作为微调数据集

<img width="985" height="517" alt="image" src="https://github.com/user-attachments/assets/dd9ccae5-2afe-4bcc-a5ea-f506944bac3a" />
<img width="978" height="554" alt="image" src="https://github.com/user-attachments/assets/d6c77768-9358-474d-bb18-107b612b9acd" />

通过运行文件data_clean.py对数据集进行清洗融合

# 五、Qwen微调流程

## 1.LoRA参数注入
```python
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
```


## 2.设置微调参数

```python
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = combined_dataset,
    eval_dataset = None,    # Can set up evaluation
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 2,    # Use GA to mimic batch size
        warmup_steps = 5,
        num_train_epochs = 1, # Set this for 1 full training runs
        # max_steps = 1,
        learning_rate = 2e-4,  # Reduce to 2e-5 for long training runs
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "wandb",    # Use this for WandB etc
    ),
```


其中SFTTrainer是一个专门为指令微调设计的训练器，封装了Hugging Face的Trainer，而SFTConfig配置训练参数的专用类，功能类似TrainingArguments。

## 3.进行微调

运行文件qwen_lora.py来进行微调
