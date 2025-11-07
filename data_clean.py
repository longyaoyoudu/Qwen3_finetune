import os

# os.environ["HTTP_PROXY"] = "http://127.0.0.1:10080"
# os.environ["HTTPS_PROXY"] = "http://127.0.0.1:10080"

from datasets import load_dataset
from unsloth import FastLanguageModel

from datasets import load_from_disk

from modelscope.msdatasets import MsDataset

reasoning_dataset = MsDataset.load('nv-community/OpenMathReasoning')
non_reasoning_dataset = MsDataset.load('mlabonne/FineTome-100k', subset_name='default', split='train')
# 使用示例
# reasoning_dataset = load_local_dataset("./correct_openmath/Dataset")


# reasoning_dataset = load_dataset("unsloth/OpenMathReasoning-mini", split="cot")
# non_reasoning_dataset = load_dataset("mlabonne/FineTome-100k", split="train")

# 在导入任何库之前设置环境变量
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["DISABLE_UNSLOTH_STATS"] = "1"  # 禁用 Unsloth 统计

# 阻止所有网络连接
import socket
original_socket = socket.socket
def guarded_socket(*args, **kwargs):
    raise RuntimeError("网络访问被阻止：强制离线模式")
socket.socket = guarded_socket

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


def generate_conversation(examples):
    problems = examples["problem"]
    solutions = examples["generated_solutions"]
    conversations = []
    for problem,solution in zip(problems, solutions):
        conversations.append([
            {"role": "user",        "content": problem},
            {"role": "assistant",   "content": solution},
        ])
    return{"conversations":conversations,}

reasoning_data = reasoning_dataset.map(generate_conversation, batched=True)

print(reasoning_data["conversations"])

reasoning_conversations = tokenizer.apply_chat_template(
    reasoning_data["conversations"],
    tokenize = False,
)

print(reasoning_conversations[0])