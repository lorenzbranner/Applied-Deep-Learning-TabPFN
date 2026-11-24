"""Step 2 of the experiment: choose the base columns of every table (PREREGISTRATION.md).

For every table in data/:
  1) a random 25 % of the rows form the selection part; they are used only here,
  2) every column gets its univariate ROC AUC on the selection part, as max(AUC, 1 - AUC),
  3) columns are taken by score, skipping a column whose |correlation| with an already
     taken one is above 0.9, until 10 columns (fewer if the table has fewer),
  4) gate: logistic regression with 50 training rows on the chosen columns, tested on
     the other rows of the selection part, mean ROC AUC over 10 random splits >= 0.5.

Writes data/selection.json (chosen columns, scores, gate, the rows of the selection
part) and prints one line per table.

    python select_columns.py
"""
import torch  # noqa: F401  keep first: on Windows torch must load before scikit-learn

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path(__file__).parent / "data"
SELECTION_SHARE = 0.25
N_COLUMNS = 10
MAX_CORRELATION = 0.9
GATE_TRAIN_ROWS, GATE_SPLITS, GATE_MIN_AUC = 50, 10, 0.5


def score(column, y):
    """How well one column alone separates the classes: 0.5 = not at all, 1 = fully."""
    if np.std(column) == 0:
        return 0.5
    auc = roc_auc_score(y, column)
    return max(auc, 1 - auc)


def choose_columns(x, y):
    scores = {c: score(x[c].values, y) for c in x.columns}
    ranked = sorted(scores, key=scores.get, reverse=True)
    correlation = x.corr().abs()
    chosen = []
    for c in ranked:
        if len(chosen) == N_COLUMNS:
            break
        if all(not correlation.loc[c, d] > MAX_CORRELATION for d in chosen):
            chosen.append(c)
    return chosen, scores


def gate(x, y):
    """Mean test ROC AUC of logistic regression with 50 training rows. A split whose
    training rows contain only one class predicts a constant (ROC AUC 0.5)."""
    aucs, one_class = [], 0
    for seed in range(GATE_SPLITS):
        order = np.random.RandomState(seed).permutation(len(y))
        train, test = order[:GATE_TRAIN_ROWS], order[GATE_TRAIN_ROWS:]
        if len(np.unique(y[train])) < 2 or len(np.unique(y[test])) < 2:
            aucs.append(0.5)
            one_class += 1
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(x[train], y[train])
        aucs.append(roc_auc_score(y[test], model.predict_proba(x[test])[:, 1]))
    return float(np.mean(aucs)), one_class


selection, lines = {}, []
for path in sorted(DATA.glob("*.csv"), key=lambda p: p.stem.lower()):
    table = pd.read_csv(path)
    x_all, y_all = table.drop(columns="target"), table.target.values
    rows = np.sort(np.random.RandomState(0).permutation(len(table))[
        :int(round(SELECTION_SHARE * len(table)))])
    x, y = x_all.iloc[rows], y_all[rows]
    chosen, scores = choose_columns(x, y)
    gate_auc, one_class = gate(x[chosen].values, y)
    selection[path.stem] = {
        "columns": chosen,
        "scores": {c: round(scores[c], 4) for c in chosen},
        "best_skipped_score": round(max([s for c, s in scores.items() if c not in chosen],
                                        default=float("nan")), 4),
        "gate_roc_auc": round(gate_auc, 4), "gate_splits_with_one_class": one_class,
        "passes_gate": gate_auc >= GATE_MIN_AUC,
        "selection_rows": rows.tolist(),
        "rows": len(table), "class_1_share": round(float(y_all.mean()), 4),
    }
    s = [scores[c] for c in chosen]
    lines.append({"table": path.stem, "rows": len(table), "columns": x_all.shape[1],
                  "chosen": len(chosen), "minority %": round(100 * min(y_all.mean(), 1 - y_all.mean()), 1),
                  "best col AUC": round(max(s), 3), "10th col AUC": round(min(s), 3),
                  "gate AUC (LR, 50 rows)": round(gate_auc, 3),
                  "one-class splits": one_class, "passes": gate_auc >= GATE_MIN_AUC})

(DATA / "selection.json").write_text(json.dumps(selection, indent=1))
print(pd.DataFrame(lines).to_string(index=False))
