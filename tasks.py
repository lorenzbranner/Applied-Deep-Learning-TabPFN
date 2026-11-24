"""Models and metrics shared by experiment.py, walkthrough.py and app.py."""
import torch  # noqa: F401  keep first: on Windows torch must load before scikit-learn

import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from model import load_model

HERE = Path(__file__).parent
CHECKPOINTS = HERE / "checkpoints"      # train_all.py writes seed1, seed2, seed3 here
# The two mini-PFNs: prior in prior.py -> name in tables and figures.
PRIORS = {"all": "Mini-PFN A", "some": "Mini-PFN B"}
OFFICIAL = "TabPFN v2 (official)"


# ------------------------------------------------------------------ models --
# Every model is a function (x_train, y_train, x_query) -> P(class 1 | x_query).
def sklearn_model(make_estimator):
    def predict(x_train, y_train, x_query):
        if len(np.unique(y_train)) < 2:      # only one class seen: nothing to fit
            return np.full(len(x_query), float(y_train[0]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # MLP convergence warnings
            return make_estimator().fit(x_train, y_train).predict_proba(x_query)[:, 1]
    return predict


BASELINES = {
    "Logistic regression": sklearn_model(
        lambda: make_pipeline(StandardScaler(), LogisticRegression())),
    "MLP (gradient descent)": sklearn_model(
        lambda: make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(64, 64), max_iter=1000, random_state=0))),
    "Gradient-boosted trees": sklearn_model(
        lambda: GradientBoostingClassifier(random_state=0)),
}


def available_steps(folder, prior):
    """Pre-training steps for which a snapshot of this prior exists."""
    return sorted(int(p.stem.split("_")[-1]) for p in Path(folder).glob(f"{prior}_[0-9]*.pt"))


def mini_pfn(folder, prior, step):
    """Our pre-trained mini-PFN; use its .predict_proba like the other models."""
    return load_model(Path(folder) / f"{prior}_{step:06d}.pt")


def official_tabpfn(x_train, y_train, x_query, n_estimators=4):
    """The pre-trained TabPFN v2 of the paper. Default: an ensemble of 4 forward
    passes on differently preprocessed copies of the table; the app uses 1."""
    from tabpfn import TabPFNClassifier
    return sklearn_model(lambda: TabPFNClassifier(
        device="cpu", random_state=0, n_estimators=n_estimators))(x_train, y_train, x_query)


# ----------------------------------------------------------------- metrics --
def accuracy(proba, y):
    return float(((proba > 0.5) == (y == 1)).mean())


def log_loss(proba, y):
    """Cross-entropy of the predicted probabilities; punishes over-confidence."""
    p = np.clip(proba, 1e-6, 1 - 1e-6)
    return float(-np.mean(np.where(y == 1, np.log(p), np.log(1 - p))))
