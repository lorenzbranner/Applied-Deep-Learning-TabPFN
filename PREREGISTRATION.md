# Pre-registration of the experiment

Fixed on 2026-11-23, before the columns were selected, before the pilot run and
before any model was evaluated on a real table.

## Question

Does a prior in which some columns are unrelated to the target (prior B) make the
mini-PFN more robust to uninformative columns than a prior in which every column
matters (prior A)?

## Tables

All binary classification tables without categorical columns and without missing
values in the paper's Extended Data Table 3 (AutoML Benchmark, at most 10,000 rows
and 500 features), downloaded from OpenML by `data/download.py`:

| Table | OpenML ID | Columns | Rows |
|---|---|---|---|
| ada | 41156 | 48 | 4147 |
| blood-transfusion-service-center | 1464 | 4 | 748 |
| kc1 | 1067 | 21 | 2109 |
| madeline | 41144 | 259 | 3140 |
| ozone-level-8hr | 1487 | 72 | 2534 |
| pc4 | 1049 | 37 | 1458 |
| philippine | 41145 | 308 | 5832 |
| phoneme | 1489 | 5 | 5404 |
| qsar-biodeg | 1494 | 41 | 1055 |
| Satellite | 40900 | 36 | 5100 |
| sylvine | 41146 | 20 | 5124 |
| wilt | 40983 | 5 | 4839 |

The two OpenML label values, sorted, become class 0 and class 1.

## Base columns (`select_columns.py`)

1. Per table, a random 25 % of the rows (numpy `RandomState(0)`) is the selection
   part. These rows are never used for training or testing.
2. Score of a column: its univariate ROC AUC on the selection part, as
   max(AUC, 1 − AUC).
3. Columns are taken by score; a column is skipped if its absolute Pearson
   correlation with a column already taken is above 0.9. Stop at 10 columns.
4. Gate: logistic regression (standardised), 50 training rows on the chosen columns,
   tested on the other rows of the selection part, mean ROC AUC over 10 random
   splits. A table below 0.5 is excluded. No table is excluded for any other reason.

Deviation from the paper, on purpose: the paper uses all columns. The best 10 give
the base table a signal that the added columns can dilute; otherwise every model
would sit at chance.

## Pre-training

- Mini-PFN (`model.py`) and prior (`prior.py`) unchanged, except that the number of
  columns is drawn log-uniformly from 2 to 50 (`train.py --max-columns 50 --columns
  log-uniform`), so that 10 base + 30 added columns lie inside the prior.
- 3 runs per prior (seeds 1, 2, 3), identical settings for A and B.
- Number of steps, set by one pilot run of prior B (seed 0, 6,000 steps, synthetic
  data only): if the mean training loss of the last 100 steps is lower by more than
  0.01 than the mean of steps 2,901 to 3,000, all runs get 5,000 steps, else 3,000.

## Evaluation

- Only on the other 75 % of the rows.
- 10 splits (seeds 0 to 9): 50 training rows, drawn stratified (minority class
  max(2, round(50 × its share)) rows), and at most 500 test rows drawn at random
  from the rest.
- Why 50 rows and not all rows as in the paper: the claim is about small data, where
  uninformative columns hurt most, and the mini-PFN saw 16 to 64 labelled rows in
  pre-training.
- Added columns: 0, 5, 10, 20, 30 shuffled copies of randomly chosen base columns,
  as in the paper (Fig. 5a).
- Models: mini-PFN A and B (3 runs each), logistic regression, MLP, gradient-boosted
  trees (`tasks.py`), TabPFN v2 (`tabpfn` 2.0.9, default ensemble of 4).
- Metric: ROC AUC (raw, not normalised), as in the paper.

## Hypothesis and test

- **H1:** at 30 added columns, mini-PFN B has a higher ROC AUC than mini-PFN A.
  Value per table: mean over 10 splits and 3 runs. Two-sided Wilcoxon signed-rank
  test over the tables that pass the gate, α = 0.05. Also reported: on how many
  tables B beats A, and the same comparison for every single run.
- Described, not tested: the gap B − A from 0 to 30 added columns, and the drop of
  TabPFN v2 compared with the drop of B.
- Every table that passes the gate is reported, also those where B does not win.

## Results

**Before the evaluation**

- Gate (`data/selection.json`): all 12 tables pass, ROC AUC 0.605 (madeline) to
  0.934 (sylvine).
- Pilot: mean loss of steps 2,901 to 3,000 = 0.4984, of steps 5,901 to 6,000 =
  0.5265. Not lower, so all runs get **3,000 steps**.

**Evaluation** (`python experiment.py --steps 3000`, 50 min; `results/`)

| Added columns | B wins (of 12) | Mean ROC AUC A | Mean ROC AUC B | Wilcoxon p |
|---:|---:|---:|---:|---:|
| 0 | 11 | 0.763 | 0.777 | 0.021 |
| 5 | 11 | 0.744 | 0.761 | 0.012 |
| 10 | 11 | 0.734 | 0.757 | 0.003 |
| 20 | 12 | 0.718 | 0.749 | 0.0005 |
| **30** | **11** | **0.708** | **0.743** | **0.0015** |

- **H1 holds:** at 30 added columns B beats A on 11 of 12 tables (exception:
  Satellite, 0.895 vs. 0.898), p = 0.0015. Each of the 3 runs on its own shows B above
  A at every number of added columns (`results/runs.csv`).
- **Limit:** B is ahead already at 0 added columns (p = 0.021). The gap grows from
  0.014 to 0.035; from 0 to 30 added columns A drops 0.055, B 0.034.
- For comparison, not tested: TabPFN v2 drops from 0.804 to 0.760, logistic
  regression from 0.794 to 0.689.
