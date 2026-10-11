"""The fresh repositories of the onboard suite: one small, realistic training
project per script variant. Each is a dict {path: content}. The scripts are
deterministic (fixed seeds, CPU torch, a few seconds), so a run of the
untouched script gives the ground truth the checker compares against.

None of them contains modgrad code or mentions modgrad: the agent has to set
it up from the user's prompt and llms.txt alone.
"""

GITIGNORE = "__pycache__/\n*.pt\nout/\n"

# ------------------------------------------------- 1. plain PyTorch loop

PLAIN_TRAIN = '''"""Fit a small MLP to a noisy sine wave."""
import argparse
import math

import torch
from torch import nn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    x = torch.linspace(-math.pi, math.pi, 1024).unsqueeze(1)
    y = torch.sin(x) + 0.05 * torch.randn_like(x)
    model = nn.Sequential(nn.Linear(1, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for step in range(1, args.steps + 1):
        idx = torch.randint(0, x.shape[0], (128,))
        loss = nn.functional.mse_loss(model(x[idx]), y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 10 == 0:
            print(f"step {step} loss {loss.item():.4f}")

    torch.save(model.state_dict(), "model.pt")


if __name__ == "__main__":
    main()
'''

PLAIN = {
    "README.md": "# sine-mlp\n\nA toy regression model. Train it with `python train.py`.\n",
    "train.py": PLAIN_TRAIN,
    ".gitignore": GITIGNORE,
}

# ------------------------------- 2. Hugging Face Trainer-style fine-tune
# transformers is importable on the bench machine but accelerate is not, so a
# real Trainer cannot run. The repo vendors a small stand-in with the same
# constructor, callback hooks and printed log dicts.

MINITRAINER = '''"""A small stand-in for transformers.Trainer: same constructor, callback hooks
(on_train_begin / on_log / on_train_end with args, state, control) and the
same printed log dicts, so this repo runs without transformers/accelerate.
Swap the import for `from transformers import Trainer, TrainingArguments`
on a machine that has them."""
import math
import time
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import torch


@dataclass
class TrainingArguments:
    output_dir: str
    num_train_epochs: float = 3.0
    per_device_train_batch_size: int = 16
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    logging_steps: int = 10
    seed: int = 42
    report_to: List[str] = field(default_factory=lambda: ["none"])
    run_name: Optional[str] = None

    def to_dict(self):
        return asdict(self)

    def to_sanitized_dict(self):
        return {k: v if isinstance(v, (int, float, str, bool)) else str(v) for k, v in asdict(self).items()}


@dataclass
class TrainerState:
    global_step: int = 0
    max_steps: int = 0
    epoch: float = 0.0
    log_history: list = field(default_factory=list)
    is_world_process_zero: bool = True
    best_metric: Optional[float] = None


@dataclass
class TrainerControl:
    should_log: bool = False


class TrainerCallback:
    def on_train_begin(self, args, state, control, **kwargs):
        pass

    def on_log(self, args, state, control, logs=None, **kwargs):
        pass

    def on_train_end(self, args, state, control, **kwargs):
        pass


class Trainer:
    def __init__(self, model, args, train_dataset, callbacks=None):
        self.model, self.args, self.train_dataset = model, args, train_dataset
        self.callbacks = list(callbacks or [])
        self.state, self.control = TrainerState(), TrainerControl()

    def add_callback(self, cb):
        self.callbacks.append(cb)

    def _event(self, name, **kw):
        for cb in self.callbacks:
            getattr(cb, name)(self.args, self.state, self.control, model=self.model, **kw)

    def log(self, logs):
        logs = {**logs, "epoch": round(self.state.epoch, 2)}
        self.state.log_history.append({**logs, "step": self.state.global_step})
        self._event("on_log", logs=logs)
        print(logs)

    def train(self):
        a = self.args
        torch.manual_seed(a.seed)
        n = len(self.train_dataset)
        bs = a.per_device_train_batch_size
        per_epoch = math.ceil(n / bs)
        self.state.max_steps = int(per_epoch * a.num_train_epochs)
        opt = torch.optim.AdamW(self.model.parameters(), lr=a.learning_rate, weight_decay=a.weight_decay)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0 - s / self.state.max_steps)
        self._event("on_train_begin")
        t0, running, count, total = time.time(), 0.0, 0, 0.0
        g = torch.Generator().manual_seed(a.seed)
        while self.state.global_step < self.state.max_steps:
            order = torch.randperm(n, generator=g)
            for i in range(0, n, bs):
                if self.state.global_step >= self.state.max_steps:
                    break
                batch = self.train_dataset[order[i:i + bs]]
                loss = self.model(**batch)["loss"]
                loss.backward()
                opt.step()
                sched.step()
                opt.zero_grad()
                self.state.global_step += 1
                self.state.epoch = self.state.global_step / per_epoch
                running += loss.item()
                count += 1
                total += loss.item()
                if self.state.global_step % a.logging_steps == 0:
                    self.log({"loss": round(running / count, 4), "learning_rate": sched.get_last_lr()[0]})
                    running, count = 0.0, 0
        rt = time.time() - t0
        self.log({"train_runtime": round(rt, 4), "train_samples_per_second": round(self.state.max_steps * bs / rt, 3),
                  "train_steps_per_second": round(self.state.max_steps / rt, 3), "train_loss": total / self.state.max_steps})
        self._event("on_train_end")
'''

HF_TRAIN = '''"""Fine-tune a tiny sentiment classifier with the Trainer API."""
import torch
from torch import nn

from minitrainer import Trainer, TrainingArguments

VOCAB, SEQ = 64, 12


class ToyDataset:
    """Token sequences; label 1 if 'good' tokens (< 16) outnumber 'bad' ones (>= 48)."""

    def __init__(self, n, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.input_ids = torch.randint(0, VOCAB, (n, SEQ), generator=g)
        good = (self.input_ids < 16).sum(1)
        bad = (self.input_ids >= 48).sum(1)
        self.labels = (good > bad).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {"input_ids": self.input_ids[idx], "labels": self.labels[idx]}


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.EmbeddingBag(VOCAB, 32)
        self.head = nn.Linear(32, 2)

    def forward(self, input_ids, labels=None):
        logits = self.head(self.emb(input_ids))
        out = {"logits": logits}
        if labels is not None:
            out["loss"] = nn.functional.cross_entropy(logits, labels)
        return out


def main():
    torch.manual_seed(0)
    args = TrainingArguments(output_dir="out/sentiment", num_train_epochs=4, per_device_train_batch_size=32,
                             learning_rate=1e-2, logging_steps=10, seed=0)
    trainer = Trainer(model=TinyClassifier(), args=args, train_dataset=ToyDataset(1600))
    trainer.train()


if __name__ == "__main__":
    main()
'''

HF = {
    "README.md": "# tiny-sentiment\n\nFine-tunes a tiny classifier with a Trainer-style loop (`minitrainer.py` mirrors the transformers Trainer API). Run `python train.py`.\n",
    "train.py": HF_TRAIN,
    "minitrainer.py": MINITRAINER,
    ".gitignore": GITIGNORE,
}

# ----------------------------------------------- 3. RL / GRPO-style loop

RL_TRAIN = '''"""GRPO on a toy task: for each of 16 prompts the policy picks one of 8
responses; a reward model scores them. Group-relative advantages, a KL
penalty to the frozen reference policy."""
import argparse

import torch
from torch import nn


def reward_fn(prompts, responses):
    target = (prompts * 3) % 8
    dist = (responses - target).abs()
    return torch.where(dist == 0, 1.0, torch.where(dist == 1, 0.3, 0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=0.02)
    args = ap.parse_args()

    torch.manual_seed(0)
    policy = nn.Embedding(16, 8)
    ref = nn.Embedding(16, 8)
    ref.load_state_dict(policy.state_dict())
    ref.requires_grad_(False)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)

    for it in range(1, args.iters + 1):
        prompts = torch.randint(0, 16, (8,)).repeat_interleave(args.group)
        logp_all = torch.log_softmax(policy(prompts), -1)
        responses = torch.multinomial(logp_all.exp().detach(), 1).squeeze(1)
        rewards = reward_fn(prompts, responses)
        r = rewards.view(-1, args.group)
        adv = ((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-4)).view(-1)
        logp = logp_all.gather(1, responses[:, None]).squeeze(1)
        ref_logp = torch.log_softmax(ref(prompts), -1)
        kl = (logp_all.exp() * (logp_all - ref_logp)).sum(-1).mean()
        loss = -(adv * logp).mean() + args.beta * kl
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 5 == 0:
            print(f"step {it}/{args.iters} | reward {rewards.mean().item():.4f} | kl {kl.item():.5f} | loss {loss.item():.4f}")


if __name__ == "__main__":
    main()
'''

RL = {
    "README.md": "# grpo-toy\n\nGRPO on a toy prompt/response task. Train with `python train_grpo.py`.\n",
    "train_grpo.py": RL_TRAIN,
    ".gitignore": GITIGNORE,
}

# --------------------------------------------- 4. a script that uses wandb

WANDB_TRAIN = '''"""Train a small MLP classifier on synthetic 2-D spirals; logs to Weights & Biases."""
import math

import torch
import wandb
from torch import nn


def spirals(n, seed):
    g = torch.Generator().manual_seed(seed)
    t = torch.rand(n, generator=g) * 3 * math.pi
    label = torch.randint(0, 2, (n,), generator=g)
    r = t / (3 * math.pi)
    ang = t + label * math.pi
    x = torch.stack([r * torch.cos(ang), r * torch.sin(ang)], 1) + 0.02 * torch.randn(n, 2, generator=g)
    return x, label


def main():
    config = {"lr": 1e-2, "steps": 200, "batch_size": 64, "hidden": 64}
    run = wandb.init(project="spirals", config=config)
    torch.manual_seed(0)
    x, y = spirals(2048, 0)
    xv, yv = spirals(512, 1)
    model = nn.Sequential(nn.Linear(2, config["hidden"]), nn.ReLU(), nn.Linear(config["hidden"], config["hidden"]), nn.ReLU(), nn.Linear(config["hidden"], 2))
    opt = torch.optim.Adam(model.parameters(), lr=config["lr"])
    for step in range(1, config["steps"] + 1):
        idx = torch.randint(0, len(x), (config["batch_size"],))
        logits = model(x[idx])
        loss = nn.functional.cross_entropy(logits, y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 10 == 0:
            acc = (logits.argmax(1) == y[idx]).float().mean().item()
            wandb.log({"train/loss": loss.item(), "train/acc": acc}, step=step)
    with torch.no_grad():
        val_acc = (model(xv).argmax(1) == yv).float().mean().item()
    wandb.log({"val/acc": val_acc}, step=config["steps"])
    run.finish()


if __name__ == "__main__":
    main()
'''

WANDB = {
    "README.md": "# spirals\n\nA small MLP on synthetic spirals; metrics go to Weights & Biases. Run `python train.py`.\n",
    "train.py": WANDB_TRAIN,
    ".gitignore": GITIGNORE + "wandb/\n",
}

# variant -> (files, the script to run, how the original reports metrics)
VARIANTS = {
    "plain": (PLAIN, "train.py"),
    "hf": (HF, "train.py"),
    "rl": (RL, "train_grpo.py"),
    "wandb": (WANDB, "train.py"),
}

# A stand-in `wandb` used only by the harness to record the ground truth of
# the wandb variant (the sandbox has no wandb, as after `pip uninstall wandb`).
WANDB_RECORDER = '''import json, os
_out = open(os.environ["WANDB_RECORD"], "w")
class _Run:
    def finish(self):
        _out.flush()
def init(**kw):
    return _Run()
def log(d, step=None, **kw):
    _out.write(json.dumps({"step": step, **d}) + "\\n")
'''
