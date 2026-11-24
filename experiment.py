"""The experiment (PREREGISTRATION.md): mini-PFN A against B on the 12 tables of the
paper's benchmark, with 0 to 30 added uninformative columns.

Needs data/*.csv (data/download.py), data/selection.json (select_columns.py) and the
pre-trained mini-PFNs in checkpoints/seed1-3 (train_all.py).

Writes into results/:
  auc.csv            one row per table, number of added columns, split, model and run
  auc_mean.csv       mean ROC AUC per table, model and number of added columns
  wilcoxon.csv       B against A over the tables, for every number of added columns
  runs.csv           the same comparison for every single pre-training run

    python experiment.py --steps 3000
"""
import torch  # noqa: F401  keep first: on Windows torch must load before scikit-learn

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

import tasks

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "results"
CHECKPOINTS = tasks.CHECKPOINTS
ADDED = [0, 5, 10, 20, 30]
SEEDS = range(10)
N_TRAIN, MAX_TEST, MIN_PER_CLASS = 50, 500, 2
A, B = tasks.PRIORS["all"], tasks.PRIORS["some"]


# ------------------------------------------------------------------ tables --
def load(table, selection):
    """The evaluation part of one table: chosen columns, rows outside the selection part."""
    frame = pd.read_csv(DATA / f"{table}.csv")
    keep = np.setdiff1d(np.arange(len(frame)), selection["selection_rows"])
    return frame[selection["columns"]].values[keep], frame.target.values[keep]


def split(y, seed, n_train=N_TRAIN):
    """Stratified training rows (minority class: max(2, round(n_train x share))),
    then at most 500 random test rows from the rest."""
    rng = np.random.RandomState(seed)
    minority = 0 if np.mean(y) > 0.5 else 1      # the rarer class (class 1 if equal)
    n_minority = max(MIN_PER_CLASS, int(round(n_train * np.mean(y == minority))))
    rows = {c: rng.permutation(np.flatnonzero(y == c)) for c in (0, 1)}
    take = {minority: n_minority, 1 - minority: n_train - n_minority}
    train = np.concatenate([rows[c][:take[c]] for c in (0, 1)])
    rest = np.concatenate([rows[c][take[c]:] for c in (0, 1)])
    test = rng.permutation(rest)[:MAX_TEST]
    return rng.permutation(train), test


def make_task(x, y, n_added, seed, n_train=N_TRAIN):
    """Uninformative columns as in the paper (Fig. 5a): shuffled copies of randomly
    chosen base columns (real values, no relation to the target). Then the same split
    for every n_added, so that only the added columns differ."""
    rng = np.random.RandomState(seed + 1000)
    added = [x[rng.permutation(len(y)), rng.randint(x.shape[1])] for _ in range(n_added)]
    x = np.column_stack([x] + added)
    train, test = split(y, seed, n_train)
    return x[train], y[train], x[test], y[test]


# ------------------------------------------------------------------ models --
def models(steps):
    folders = sorted(f for f in CHECKPOINTS.glob("seed*") if f.is_dir())
    assert len(folders) == 3, folders
    out = {name: [tasks.mini_pfn(f, prior, steps).predict_proba for f in folders]
           for prior, name in tasks.PRIORS.items()}
    out[tasks.OFFICIAL] = [tasks.official_tabpfn]          # default ensemble of 4
    out.update({name: [predict] for name, predict in tasks.BASELINES.items()})
    return out


def roc_auc(proba, y):
    return float(roc_auc_score(y, proba)) if len(np.unique(y)) == 2 else float("nan")


def main(steps):
    OUT.mkdir(exist_ok=True)
    selection = json.loads((DATA / "selection.json").read_text())
    tables = [t for t, s in selection.items() if s["passes_gate"]]
    predictors, rows, start = models(steps), [], time.time()
    for table in tables:
        x, y = load(table, selection[table])
        for n_added in ADDED:
            for seed in SEEDS:
                x_train, y_train, x_test, y_test = make_task(x, y, n_added, seed)
                for name, predicts in predictors.items():
                    for run, predict in enumerate(predicts):
                        proba = predict(x_train, y_train, x_test)
                        rows.append({"table": table, "n_added": n_added, "seed": seed,
                                     "model": name, "run": run,
                                     "roc_auc": roc_auc(proba, y_test),
                                     "accuracy": tasks.accuracy(proba, y_test),
                                     "log_loss": tasks.log_loss(proba, y_test)})
            print(f"{table:34s} {n_added:2d} added  {time.time() - start:6.0f} s", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "auc.csv", index=False)   # partial results survive

    results = pd.DataFrame(rows)
    results.to_csv(OUT / "auc.csv", index=False)
    per_table = results.groupby(["table", "model", "n_added"]).roc_auc.mean().unstack("n_added")
    per_table.round(4).to_csv(OUT / "auc_mean.csv")

    # Main test (H1 at 30 added columns; the other columns described): one value per
    # table = mean over 10 splits and 3 runs; two-sided Wilcoxon over the tables.
    tests = []
    for n_added in ADDED:
        a, b = per_table.loc[(slice(None), A), n_added], per_table.loc[(slice(None), B), n_added]
        a, b = a.droplevel("model"), b.droplevel("model")
        tests.append({"n_added": n_added, "tables": len(a), "B wins": int((b > a).sum()),
                      "mean A": a.mean(), "mean B": b.mean(),
                      "wilcoxon p": wilcoxon(b, a).pvalue if (b != a).any() else 1.0})
    pd.DataFrame(tests).round(5).to_csv(OUT / "wilcoxon.csv", index=False)

    # Every pre-training run on its own: run k of B against run k of A.
    runs = (results[results.model.isin([A, B])]
            .groupby(["model", "run", "table", "n_added"]).roc_auc.mean()
            .groupby(["model", "run", "n_added"]).mean().unstack("n_added"))
    runs.round(4).to_csv(OUT / "runs.csv")
    print(pd.DataFrame(tests).round(4).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, required=True,
                        help="pre-training steps of the checkpoints (pilot rule: 3000 or 5000)")
    main(parser.parse_args().steps)
