"""Step 1 of the experiment: download the 12 tables from OpenML once and save them as CSV.

The tables are the binary, purely numerical classification tables of the paper's
Extended Data Table 3 (see PREREGISTRATION.md). After this script has run, the
demo needs no internet:  python data/download.py

Every CSV has the feature columns and a last column "target" with classes 0 and 1
(the two OpenML label values, sorted).
"""
from pathlib import Path

import torch  # noqa: F401  keep first: on Windows torch must load before scikit-learn

import numpy as np
from sklearn.datasets import fetch_openml

TABLES = {
    "ada": 41156, "blood-transfusion-service-center": 1464, "kc1": 1067,
    "madeline": 41144, "ozone-level-8hr": 1487, "pc4": 1049, "philippine": 41145,
    "phoneme": 1489, "qsar-biodeg": 1494, "Satellite": 40900, "sylvine": 41146,
    "wilt": 40983,
}
HERE = Path(__file__).parent

for name, did in TABLES.items():
    # sklearn's loader instead of the openml package: plain HTTP, no extra dependency.
    bunch = fetch_openml(data_id=did, as_frame=True, parser="auto")
    x, y = bunch.data, bunch.target
    assert all(t.kind in "fiu" for t in x.dtypes), f"{name}: non-numerical columns"
    assert not x.isna().any().any(), f"{name}: missing values"
    labels = sorted(np.unique(y.astype(str)))
    assert len(labels) == 2, f"{name}: {len(labels)} classes"
    x = x.astype(float)
    x["target"] = (y.astype(str) == labels[1]).astype(int).values
    x.to_csv(HERE / f"{name}.csv", index=False)
    print(f"{name:34s} id {did:6d}  {x.shape[0]:5d} rows  {x.shape[1] - 1:4d} columns  "
          f"class 1 = {labels[1]!r} ({x.target.mean():.0%})")
