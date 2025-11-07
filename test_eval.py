from evalscope import TaskConfig, run_task
import logging

logging.basicConfig(level=logging.INFO)

task_cfg = TaskConfig(
    model='./Qwen3-32B-unsloth-bnb-4bit',
    api_url='http://127.0.0.1:8000/v1/chat/completions',
    eval_type='openai_api',
    datasets=['mmlu'],  # 使用内置的MMLU基准，而不是data_collection
    dataset_args={
        'mmlu': {
            'split': 'test',
            'limit': 5,
            # 可能不需要指定dataset_id，因为内置基准已经知道数据集
        }
    },
    eval_batch_size=4,
    generation_config={
        'max_tokens': 1024,
        'temperature': 0.6,
        'top_p': 0.9,
        'top_k': 20,
        'n': 1
    },
    limit=5,
)

run_task(task_cfg=task_cfg)