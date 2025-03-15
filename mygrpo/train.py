import os
import re

from dataclasses import dataclass, fields
from typing import Optional

import torch
import torch.optim as optim

import wandb
from torch.nn.utils import clip_grad_norm_
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model_id = "meta-llama/Llama-3.2-1B-Instruct"
wandb_project = "my-tiny-grpo"  # "tiny_grpo"
wandb.init(project=wandb_project)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map=device,
)
model_ref = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map=device,
)
model_ref.eval()

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


@torch.no_grad()
def rollout(model, tokenizer, question: str, oracle_answer: str, n_rollout=12):
    model.eval()
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
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,  # why need padding, padding_size, return_attention_mask?
        padding_side="left",
        return_attention_mask=True,
    ).to(device)
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
    print(f"in rollout: {log_probs.max()=} {log_probs_ref.max()=}")

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


def approx_kl_divergence(
    log_probs: torch.Tensor,
    log_probs_ref: torch.Tensor,
    action_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Monte-Carlo approximation of KL divergence, k3 estimator, see: http://joschu.net/blog/kl-approx.html
    """

    log_ratio = log_probs_ref.float() - log_probs.float()
    if action_mask is not None:
        log_ratio = log_ratio * action_mask

    return log_ratio.exp() - log_ratio - 1


def main():
    lr = 5e-6
    optimizer = optim.Adam(model.parameters(), lr=lr)
    n_steps = 100
    n_epochs_per_step = 1

    max_norm = 1.0  # gradient clipping
    clip_eps = 0.2
    kl_weight = 0.01

    for i in range(n_steps):
        experiences = rollout(model, tokenizer, "213 + 215 =", "428")
        episode_return_sum = sum([e.reward for e in experiences])

        model.train()
        for step_epoch in range(n_epochs_per_step):
            print(f"============={step_epoch=}==============")
            sequence = torch.stack([e.sequence for e in experiences])
            log_probs = seq_log_probs(model, sequence)

            log_probs_old = torch.stack([e.log_prob for e in experiences])
            log_probs_old_ref = torch.stack([e.log_prob_ref for e in experiences])
            advantages = (
                torch.stack([e.advantage for e in experiences])
                .unsqueeze(dim=-1)
                .to(device)
            )
            print(
                f"{log_probs.max()=} {log_probs_old.max()=} {log_probs_old_ref.max()=}"
            )

            # kl = torch.nn.functional.kl_div(
            #     log_probs, log_probs_old_ref, reduction="batchmean"
            # ).mean()
            kl = approx_kl_divergence(
                log_probs=log_probs,
                log_probs_ref=log_probs_old_ref,
                action_mask=None,
            ).mean()

            assert torch.allclose(log_probs, log_probs_old, atol=1e-3, rtol=1e-3)  # if n_epoch_per_step == 1, this thould be true  # fmt: skip

            ratio = (log_probs - log_probs_old).exp()
            surr1 = ratio * advantages
            surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantages
            loss = -torch.min(surr1, surr2) + kl_weight * kl
            loss = loss.mean()
            print(f"{loss=}, {kl=}")

            if not loss.isfinite():
                print(f"Loss not finite, skipping backward, loss={loss}")
                continue

            loss.backward()
            grad_norm = clip_grad_norm_(model.parameters(), max_norm=max_norm)
            print(f"{step_epoch}: kl={kl: .4f}, grad_norm={grad_norm: .4f}")

            wandb.log({"loss": loss, "kl": kl, "grad_norm": grad_norm})

            # for p in model.parameters():
            #     if not p.grad.max().isfinite():
            #         print("p.grad is not finite")
            #         raise ValueError("p.grad is not finite")
            optimizer.step()
            # for p in model.parameters():
            #     if not p.max().isfinite():
            #         print("p is not finite")

        wandb.log({"returns": episode_return_sum})
        print(f"{i=} {episode_return_sum=}")


main()
