import re

from dataclasses import dataclass, fields
from typing import Optional

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
model_ref = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map=device,
)

system_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>
"""


@dataclass
class MyExperience:  # this represent a single experience
    sequence: torch.Tensor  # [1, seq_len]                    # used for training
    log_prob: Optional[torch.Tensor]  # [1, seq_len]          # used for training
    log_prob_ref: Optional[torch.Tensor]  # [1, seq_len]       # used for training

    reward: torch.Tensor  # [1]                                # used for training
    advantage: torch.Tensor  # [1]                             # used for training
    response: str  # not used for logging


def batch_advantage(returns):
    return (returns - returns.mean()) / (returns.std() + 0.000001)


def seq_log_probs(model, sequence):
    pad_token_id = tokenizer.eos_token_id
    attention_mask = sequence != pad_token_id
    logits = model(
        sequence, attention_mask=attention_mask
    ).logits  # [12, seq_len, vocab_size]
    print(f"{logits.shape=}")

    log_probs = logits.log_softmax(dim=-1).gather(
        dim=-1, index=sequence.unsqueeze(dim=-1)
    )
    return log_probs.squeeze(dim=-1)  # [12, seq_len]


def rollout(model, tokenizer, question: str, oracle_answer: str, n_rollout=12):
    # You are a a helpful calculator!
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {"role": "user", "content": question},
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print(f"{prompt=}")

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    print(f"{inputs=}")
    inputs["input_ids"] = inputs["input_ids"].repeat(n_rollout, 1)
    inputs["attention_mask"] = inputs["attention_mask"].repeat(n_rollout, 1)

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
    output = model.generate(
        **inputs,
        generation_config=generation_config,
        # output_scores=True,
        # return_dict_in_generate=True,
    )
    sequence_ids = output
    response = tokenizer.batch_decode(
        sequence_ids[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    returns = torch.tensor([reward_fn(i, oracle_answer) for i in response])
    advanteges = batch_advantage(returns)

    log_probs = seq_log_probs(model, sequence_ids)
    print(log_probs.shape)
    log_probs_ref = seq_log_probs(model_ref, sequence_ids)

    print(returns, advanteges)
    experiences = []
    for i in range(n_rollout):
        e = MyExperience(
            sequence=sequence_ids[i],
            log_prob=log_probs[i],
            log_prob_ref=log_probs_ref[i],  # TODO: fixme
            reward=returns[i],
            advantage=advanteges[i],
            response=response[i],
        )
        experiences.append(e)
    return experiences


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


experiences = rollout(model, tokenizer, "213 + 215 =", "428")
for e in experiences:
    print(
        f"============ {e.reward=}",
        e.response,
    )
