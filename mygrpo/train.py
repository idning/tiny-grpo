import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model_id = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map=device,
)

system_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>
"""
# You are a a helpful calculator!
messages = [
    {
        "role": "system",
        "content": system_prompt,
    },
    {"role": "user", "content": "213 + 215 ="},
]

prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
print(f"{prompt=}")

# Tokenize input
inputs = tokenizer(prompt, return_tensors="pt").to(device)
print(f"{inputs=}")
inputs["input_ids"] = inputs["input_ids"].repeat(12, 1)
inputs["attention_mask"] = inputs["attention_mask"].repeat(12, 1)

max_length = 1024
top_p = 1.0
temperature = 1.0
pad_token_id = tokenizer.eos_token_id
print(f"{pad_token_id=}")

generation_config = GenerationConfig(
    do_sample=True,
    top_p=top_p,
    temperature=temperature,
    max_length=max_length,
    pad_token_id=pad_token_id,
)
# Generate output
sequence_ids = model.generate(**inputs, generation_config=generation_config)
print(f"{sequence_ids=}")
# Decode response
response = tokenizer.batch_decode(
    sequence_ids[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
)


def reward_fn(answer: str, oracle_answer: str):
    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, answer, re.DOTALL)
    if not match:
        return 0
    if oracle_answer == match.group(1).strip():
        return 1
    if oracle_answer in match.group(1):
        return 0.5
    return 0.01


# for i in response:
#     print(i, )
returns = [reward_fn(i, "428") for i in response]
print(returns)
