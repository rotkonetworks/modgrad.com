#!/usr/bin/env python3
"""Synthetic training loop for dogfooding `obs run`: prints a decreasing
loss in a few common log formats, plus prose the parser must ignore.

    obs run -- python3 scripts/obs_fake_train.py --steps 300 --delay 0.1
"""
import argparse, math, random, sys, time

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=300)
p.add_argument("--delay", type=float, default=0.1)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
rng = random.Random(a.seed)

print("loading dataset: 12000 examples")
print("model has 1.3M parameters")
loss = 4.0
for step in range(1, a.steps + 1):
    loss = 0.3 + 3.7 * math.exp(-step / (a.steps / 4)) + rng.gauss(0, 0.05)
    lr = 3e-4 * min(1.0, step / 20) * 0.5 * (1 + math.cos(math.pi * step / a.steps))
    if step % 5 == 0:
        print(f"step {step}: loss {loss:.4f} acc {100 * (1 - loss / 4.3):.1f}% lr {lr:.2e}")
    if step % 25 == 0:
        print(f"iter {step} | val_loss={loss + 0.1 + rng.gauss(0, 0.03):.4f} | grad_norm {abs(rng.gauss(1, 0.2)):.3f}")
    if step % 100 == 0:
        print(f"[epoch {step // 100}/{a.steps // 100}] train_loss: {loss:.4f}", file=sys.stderr)
        print(f"saving checkpoint to ckpt_{step}.pt")
    time.sleep(a.delay)
print("done")
