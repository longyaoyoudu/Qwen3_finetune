from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key = openai_api_key,
    base_url = openai_api_base,
)

messages = [
    {"role":"user", "content":"你好，好久不见！"}
]

response = client.chat.completions.create(
    model = "./Qwen3-32B-unsloth-bnb-4bit",
    messages = messages,
)

print(response.choices[0].message.content)