# Qwen3_finetune
环境准备
Unsloth
vLLM
EvalScope
wandb
一、模型下载
通过魔搭社区下载 Qwen3-unsloth-bnb-4bit模型
<img width="819" height="406" alt="image" src="https://github.com/user-attachments/assets/c44eca74-5d4c-4b98-a8a5-e807d005dfdc" />
二、模型简单调用
使用vLLM对下载好的模型进行简单调用
vllm serve ./Qwen3-32B-unsloth-bnb-4bit --enable-auto-tool-choice --tool-call-parser hermes --gpu-memory-utilization 0.7
<img width="866" height="417" alt="image" src="https://github.com/user-attachments/assets/bd2a8d5a-5377-47d6-8d3e-d240b3987767" />
根据gpu显存大小自定义选择gpu内存利用率，这里以英伟达A800显卡为例
<img width="1226" height="188" alt="image" src="https://github.com/user-attachments/assets/e89bf9ae-af65-4726-bede-9d90d9aa6d4c" />
三、模型评估
使用EvalScope对模型进行压力测试
evalscope perf --url "http://127.0.0.1:8000/v1/chat/completions" --parallel 5 --model ./Qwen3-32B-unsloth-bnb-4bit --number 20 --api openai --dataset openqa --stream
<img width="998" height="302" alt="image" src="https://github.com/user-attachments/assets/ec2c177f-25b4-419e-98fa-cef7a9451a24" />
通过运行文件对模型进行压力测试

四、微调数据集准备
本项目选择使用OpenMathReasoning和FineTome-100k数据集作为微调数据集
<img width="985" height="517" alt="image" src="https://github.com/user-attachments/assets/dd9ccae5-2afe-4bcc-a5ea-f506944bac3a" />
<img width="978" height="554" alt="image" src="https://github.com/user-attachments/assets/d6c77768-9358-474d-bb18-107b612b9acd" />
通过运行文件对数据集进行清洗融合
