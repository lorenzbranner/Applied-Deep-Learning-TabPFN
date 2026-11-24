"""Check of walkthrough.py, not part of the demo:
  1) trace() gives the same probabilities as model.predict_proba,
  2) one pass over many query rows equals one pass per row (test rows never read
     each other), so the decision surface may be computed in one pass,
  3) every attention distribution sums to 1,
  4) draw_table() draws exactly the table of prior.sample_batch (same seed), also
     when the sliders of the app are set to the drawn values.

    python check_walkthrough.py
"""
import torch  # noqa: F401  keep first: on Windows torch must load before scikit-learn

import numpy as np

import tasks
import walkthrough as w

x_train, y_train = w.moons()
queries = np.random.RandomState(0).uniform([-1.5, -1], [2.5, 1.5], size=(20, 2))
run = w.walk_run()                       # the run the tab shows
print("run:", run)
for step in tasks.available_steps(run, "some"):
    model = tasks.mini_pfn(run, "some", step)
    t = w.trace(model, x_train, y_train, queries)
    together = model.predict_proba(x_train, y_train, queries)
    alone = np.array([model.predict_proba(x_train, y_train, q[None])[0] for q in queries])
    assert np.allclose(t["proba"], together, atol=1e-5), "trace differs from the model"
    assert np.allclose(together, alone, atol=1e-5), "query rows influence each other"
    for layer in t["layers"]:
        assert np.isclose(layer["feature"].sum(), 1, atol=1e-5)
        assert np.allclose(layer["sample"].sum(1), 1, atol=1e-5)
    print(f"step {step:5d}: trace = model (max diff {np.abs(t['proba'] - together).max():.1e}), "
          f"together = alone (max diff {np.abs(together - alone).max():.1e})")
from prior import sample_batch
SETTINGS = ["d", "n_train", "depth", "width", "weight_scale", "noise_std", "normal"]
for max_columns, columns in ((50, "log-uniform"), (50, "uniform")):   # both rules of prior.py
    for prior in ("all", "some"):
        for seed in range(100):
            t = w.draw_table(prior, seed, max_columns, columns)
            torch.manual_seed(seed)
            x, y, n_train = sample_batch(1, prior, max_columns, columns)
            assert np.array_equal(t["x"], x[0].numpy()) and np.array_equal(t["y"], y[0].numpy())
            assert t["n_train"] == n_train and t["used"].sum() == t["k"]
            # the sliders of the app set to the drawn values: the same table again
            choose = {name: t[name] for name in SETTINGS}
            choose["activations"] = [layer["activation"] for layer in t["layers"]]
            if prior == "some":
                choose["unrelated"] = t["d"] - t["k"]
            again = w.draw_table(prior, seed, max_columns, columns, choose)
            assert np.array_equal(again["x"], t["x"]) and np.array_equal(again["y"], t["y"])
print("draw_table = prior.sample_batch for 2 column rules x 2 priors x 100 seeds, "
      "also with every choice set to the drawn value")
print("all checks passed")
