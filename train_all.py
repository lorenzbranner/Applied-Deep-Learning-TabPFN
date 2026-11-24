"""Step 3 of the experiment: pre-train the 6 mini-PFNs one after another
(priors A and B, seeds 1 to 3, 2 to 50 columns drawn log-uniformly, 3,000 steps as
decided by the pilot run, see PREREGISTRATION.md). About 0.6 s per step on a laptop
CPU, so about 3 hours. A run whose history file already exists is skipped, so the
script can be restarted after an interruption.

    python train_all.py
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
STEPS = 3000

for seed in (1, 2, 3):
    for prior in ("all", "some"):
        out = HERE / "checkpoints" / f"seed{seed}"
        if (out / f"{prior}_history.json").exists():
            print(f"seed {seed} prior {prior}: done before, skipped", flush=True)
            continue
        out.mkdir(parents=True, exist_ok=True)
        print(f"seed {seed} prior {prior}: training ...", flush=True)
        with open(out / f"{prior}_train.log", "w") as log:
            subprocess.run([sys.executable, "-u", str(HERE / "train.py"), "--prior", prior,
                            "--seed", str(seed), "--steps", str(STEPS),
                            "--max-columns", "50", "--columns", "log-uniform",
                            "--out", f"checkpoints/seed{seed}"], cwd=HERE, stdout=log, stderr=subprocess.STDOUT,
                           check=True)
print("all 6 runs done", flush=True)
