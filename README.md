# Can a prior teach a network to ignore uninformative columns?

Demo for a paper lecture on Hollmann et al., *Accurate predictions on small data
with a tabular foundation model*, Nature 637 (2025).

The same small network (mini-PFN) is pre-trained on two priors: in prior A every
column influences the target, in prior B some columns do not. Both are tested on the
12 binary, purely numerical tables of the paper's benchmark, with 0 to 30 added
shuffled columns, against TabPFN v2, logistic regression, an MLP and
gradient-boosted trees. Plan and test were fixed before the run:
[PREREGISTRATION.md](PREREGISTRATION.md).

## Run

1. Python **3.10** (tested: 3.10.0, Windows 11, CPU only, no GPU needed).
2. `pip install -r requirements.txt`
3. `streamlit run app.py`, then open `http://localhost:8501`.

The first start needs internet once: `tabpfn` downloads the TabPFN v2 checkpoint.
Everything else is in the repository.

| Tab | What it does | Time (laptop CPU) |
|---|---|---|
| Live | Choose a table (default qsar-biodeg), the number of added columns (0 to 30) and of training rows (10 to 200). Computed live: mini-PFN A and B (forward pass only), TabPFN v2 (1 ensemble member) and the three baselines; ROC AUC over 5 splits. | first load about 18 s, then 10 to 25 s per change |
| Experiments | The full experiment, precomputed (`results/`). | instant |
| Walkthrough | From the prior to one prediction, step by step (moons). | 3 to 7 s |

## Precomputed, and how to reproduce it

All outputs are included, so the app needs none of these steps.

| Step | Command | Output | Time |
|---|---|---|---|
| 1. Download the 12 tables from OpenML | `python data/download.py` | `data/*.csv` | a few minutes |
| 2. Choose the base columns | `python select_columns.py` | `data/selection.json` | under 1 min |
| 3. Pre-train 6 mini-PFNs (A, B × 3 seeds, 3,000 steps) | `python train_all.py` | `checkpoints/seed1-3/` | about 3 h |
| 4. Evaluate all models | `python experiment.py --steps 3000` | `results/` | about 50 min |

The number of steps was set by a pilot run (`checkpoints/pilot_seed0/`, see
PREREGISTRATION.md).

**Result:** at 30 added columns, mean ROC AUC A 0.708, B 0.743; B wins on 11 of 12
tables, Wilcoxon p = 0.0015. B is ahead already at 0 added columns (0.763 vs. 0.777).

## Files

| File | Content |
|---|---|
| `prior.py` | Priors A and B: synthetic tables from a random MLP |
| `model.py` | The mini-PFN: one token per cell, attention across a row, then across a column |
| `train.py`, `train_all.py` | Pre-training of one mini-PFN, and of all six |
| `data/download.py`, `select_columns.py`, `experiment.py` | Tables, column selection, evaluation |
| `tasks.py` | Baselines, TabPFN v2, loading of mini-PFNs |
| `app.py`, `walkthrough.py`, `style.py`, `.streamlit/`, `static/` | The app |
| `check_app.py`, `check_walkthrough.py` | Tests of the app (response times in `results/app_timing.txt`) and of the walkthrough |
