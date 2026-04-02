# Qwen3-32B LoRA Fine-tuning Project

> 基于 Qwen3-32B + Unsloth 4bit 量化 + LoRA 高效微调实战，支持 MMLU 基准评测与 vLLM 部署推理

[![Unsloth](https://img.shields.io/badge/Unsloth-4bit%20Quantization-orange)](https://unsloth.ai)
[![Qwen3](https://img.shields.io/badge/Qwen-Qwen3--32B-blue)](https://github.com/QwenLM)
[![LoRA](https://img.shields.io/badge/LoRA-Rank%2032-green)](https://arxiv.org/abs/2106.09685)
[![MMLU](https://img.shields.io/badge/MMLU-80%25%20Accuracy-purple)](https://arxiv.org/abs/2009.03300)
[![License](https://img.shields.io/badge/License-Apache--2.0-yellow)](LICENSE)

---

## 📌 项目概述

本项目对 **Qwen3-32B**（通义千问团队开源的 320 亿参数大语言模型）进行高效微调与评测：

- **LoRA 高效微调** — 使用 Unsloth + BitsAndBytes NF4 4bit 量化，6×A800 80GB 可运行
- **工具调用实战** — Qwen3 原生工具调用（Tool Call）示例
- **MMLU 基准评测** — 在 285 题 MMLU 数据集上达到 **80% 准确率**
- **vLLM 部署推理** — OpenAI-API 兼容接口，55+ tokens/s 输出吞吐量
- **EvalScope 评测框架** — 自动化模型评测工具

---

## 🗂️ 项目结构

```
qwen_program/
├── qwen_lora.py                 # LoRA 微调主脚本
├── qwen_weather.py               # Qwen3 工具调用示例
├── download_dataset.py           # HuggingFace 数据集下载
├── test_evalscope.py            # EvalScope 评测脚本
├── test_modelscopedataset.py    # ModelScope 数据集评测
├── test_vllmapi.py             # vLLM API 连接测试
│
├── data/
│   └── cleaned_qwen3_dataset/   # 清洗后的 Qwen3 训练数据集
│
├── outputs/                     # 评测输出
│   └── [timestamp]/
│       ├── reports/             # MMLU 评测报告
│       ├── reviews/             # 各子任务详细结果
│       └── benchmark_*.json     # vLLM 吞吐基准数据
│
├── wandb/                       # WandB 实验日志
└── README.md
```

---

## 🧠 模型配置

| 参数 | 值 |
|------|-----|
| **模型** | Qwen3-32B (Unsloth 4bit NF4) |
| **参数量** | 33.03B |
| **隐藏层维度** | 5120 |
| **层数** | 64 |
| **注意力头数** | 64 |
| **KV 头数** | 8 (GQA) |
| **上下文长度** | 40,960 |
| **最大序列长度** | 8,192 |
| **量化方式** | BitsAndBytes NF4 (4bit) |
| **训练精度** | BF16 |

---

## 🚀 快速开始

### 环境依赖

```bash
pip install unsloth transformers trl bitsandbytes
pip install wandb swanlab evalscope vllm
pip install datasets accelerate
```

### 1. LoRA 微调

```bash
# 单卡运行 Qwen3-32B 4bit LoRA 微调
python qwen_lora.py
```

**核心训练参数：**

| 参数 | 值 |
|------|-----|
| LoRA Rank | 32 |
| LoRA Alpha | 32 |
| Target Modules | q/k/v/o proj + gate/up/down proj |
| Batch Size (per GPU) | 2 |
| Gradient Accumulation | 4 |
| Max Seq Length | 8192 |
| Learning Rate | 2e-4 |
| Optimizer | AdamW 8bit |
| Training Steps | 30 |
| 显存占用 | ~A800 80GB 单卡可运行 |

### 2. 工具调用示例

```bash
# 测试 Qwen3 原生工具调用能力
python qwen_weather.py
```

使用 chat template 的 `tools` 参数注册工具，模型自动生成 tool_calls。

### 3. 模型评测

```bash
# 启动 vLLM 服务
vllm serve Qwen3-32B-unsloth-bnb-4bit --api-url http://127.0.0.1:8000

# 运行 MMLU 评测
python test_evalscope.py
```

---

## 📊 MMLU 评测结果

基于 Qwen3-32B 4bit 量化模型在 MMLU 基准上的表现：

| 领域 | 准确率 |
|------|--------|
| **Social Sciences** | **86.7%** |
| **Other** | **80.0%** |
| **Humanities** | **83.1%** |
| **STEM** | **73.3%** |
| **Overall** | **80.0%** |

**亮点子任务（满分）：**
- 高中世界历史 / 欧洲历史 / 美国历史 ✅
- 哲学、形式逻辑、道德场景 ✅
- 线性代数、高中生物/化学/计算机 ✅
- 机器学习、电气工程、初等数学 ✅

---

## ⚡ vLLM 部署性能

| 指标 | 值 |
|------|-----|
| **输出吞吐** | 55.5 tokens/s |
| **总吞吐** | 56.7 tokens/s |
| **请求吞吐** | 0.04 req/s |
| **平均延迟** | 115.9 s |
| **首 Token 延迟** | 0.21 s |
| **单 Token 延迟** | 0.087 s |

**测试配置：** 6× NVIDIA A800 80GB PCIe，Concurrency=5

---

## 🔧 核心技术

### 1. Unsloth 4bit 量化
使用 BitsAndBytes NF4 量化格式，将 32B 模型的显存需求从 ~256GB 压缩到单卡可运行范围：

```python
load_in_4bit = True
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen3-32B-unsloth-bnb-4bit",
    max_seq_length=8192,
    dtype=None,
    load_in_4bit=load_in_4bit,
)
```

### 2. LoRA 高效微调
仅训练 0.1% 参数，大幅降低显存和计算成本：

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    use_gradient_checkpointing="unsloth",
)
```

### 3. Unsloth 梯度检查点
比常规实现节省 30% 显存，支持更大 batch size：

```python
use_gradient_checkpointing = "unsloth"  # vs True
```

---

## 📋 工具脚本说明

| 脚本 | 功能 |
|------|------|
| `qwen_lora.py` | Qwen3-32B 4bit LoRA 微调主脚本 |
| `qwen_weather.py` | Qwen3 工具调用实战示例 |
| `download_dataset.py` | HuggingFace 数据集下载 |
| `test_evalscope.py` | EvalScope 框架 MMLU 评测 |
| `test_modelscopedataset.py` | ModelScope 数据集评测 |
| `test_vllmapi.py` | vLLM API 连接验证 |

---

## 🙏 致谢

- **Qwen Team** — 开源 Qwen3 系列模型 [@GitHub](https://github.com/QwenLM)
- **Unsloth** — 高效 LLM 微调框架 [@unsloth.ai](https://unsloth.ai)
- **EvalScope** — 自动化模型评测框架 [@ModelScope](https://modelscope.cn)
- **TRL** — HuggingFace 强化学习与微调工具 [@GitHub](https://github.com/huggingface/trl)

---

## 📝 License

Apache License 2.0
