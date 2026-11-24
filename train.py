"""Pre-train the mini-PFN on synthetic tables drawn from one prior.

Every training step
  1) draws 32 brand-new synthetic tables from the prior,
  2) hides the labels of the last rows of each table (the held-out rows),
  3) lets the transformer predict them from the labelled rows in one forward pass,
  4) minimises the cross-entropy of those predictions.

This is the paper's loss  L = E[ -log q(y_test | x_test, D_train) ]  (Methods,
"Training details"). No real data is involved, and because every batch is new
the model cannot overfit: the training loss is also a validation loss.

Both priors are trained with exactly the same settings and seed (train_all.py
runs all six: priors A and B, seeds 1 to 3):
    python train.py --prior all  --seed 1 --out checkpoints/seed1
    python train.py --prior some --seed 1 --out checkpoints/seed1
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from model import MiniPFN
from prior import sample_batch

parser = argparse.ArgumentParser()
parser.add_argument("--prior", required=True, choices=["all", "some"])
parser.add_argument("--steps", type=int, default=3000)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--out", default="checkpoints/seed1")
parser.add_argument("--max-columns", type=int, default=50,
                    help="columns per synthetic table: 2 to this")
parser.add_argument("--columns", default="log-uniform", choices=["uniform", "log-uniform"],
                    help="how the number of columns is drawn")
args = parser.parse_args()

BATCH_SIZE = 32      # synthetic tables per step
LR = 1e-3
# Steps at which a snapshot is saved, so the app can show the learning progress.
SNAPSHOTS = [s for s in (0, 100, 300, 1000, 3000) if s < args.steps] + [args.steps]

torch.manual_seed(args.seed)
model = MiniPFN()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)


def learning_rate(step, warmup=500):
    """Linear warm-up, then cosine annealing (as in the paper)."""
    if step < warmup:
        return LR * step / warmup
    return LR * 0.5 * (1 + math.cos(math.pi * (step - warmup) / (args.steps - warmup)))


def save(step):
    torch.save({"config": model.config, "state_dict": model.state_dict(),
                "prior": args.prior, "step": step, "batch_size": BATCH_SIZE,
                "max_columns": args.max_columns, "columns": args.columns},
               out / f"{args.prior}_{step:06d}.pt")


print(f"prior={args.prior}  parameters={sum(p.numel() for p in model.parameters()):,}")
history, running, start = [], [], time.time()
save(0)
for step in range(1, args.steps + 1):
    x, y, n_train = sample_batch(BATCH_SIZE, args.prior, args.max_columns, args.columns)
    logits = model(x, y[:, :n_train])                      # (B, held-out rows, 2)
    loss = F.cross_entropy(logits.flatten(0, 1), y[:, n_train:].flatten())

    for group in optimizer.param_groups:
        group["lr"] = learning_rate(step)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    running.append(loss.item())
    if step % 100 == 0:
        history.append({"step": step, "loss": sum(running) / len(running)})
        running = []
        print(f"step {step:6d}  loss {history[-1]['loss']:.4f}  "
              f"{time.time() - start:6.0f} s", flush=True)
    if step in SNAPSHOTS:
        save(step)

(out / f"{args.prior}_history.json").write_text(json.dumps(
    {"args": vars(args), "batch_size": BATCH_SIZE, "seconds": time.time() - start,
     "history": history}, indent=1))
print(f"done in {time.time() - start:.0f} s")
