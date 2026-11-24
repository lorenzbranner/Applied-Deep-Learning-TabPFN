"""The prior: a generator of random synthetic classification tables.

The mini-PFN never sees real data in pre-training. It sees tables drawn from
this generator, and what these tables contain is what it learns to handle.

One table = one random MLP (as in the paper's prior, simplified):
  - the input columns are the root causes (the inputs of the network),
  - every hidden unit is a node of the causal graph, with Gaussian noise,
  - the target is one output node, split at its median into two classes.

Two priors that differ in one property only:
  "all"   every one of the d columns is an input of the MLP.
  "some"  only k of the d columns (k drawn from 1 to d) are inputs of the MLP.
          The other d - k columns are drawn from the same distribution but
          never touch the target. Then the column order is shuffled, so the
          position of a column says nothing about whether it matters.

Paper (Methods, "Graph structure sampling"): disjoint subgraphs of the causal
graph "lead to features that are marginally independent of the target",
"reflecting real-world scenarios with uninformative predictors". Prior "some"
is our small version of that; prior "all" removes it.
"""
import math

import torch

# Paper: activations are "randomly sampled from a set". We use six of them.
ACTIVATIONS = [torch.tanh, torch.relu, torch.sin, torch.abs, torch.square, torch.sign]
N_HELD_OUT = 32    # rows per table whose label the model must predict in pre-training


def randint(low, high):
    """Random integer in [low, high], both ends included."""
    return int(torch.randint(low, high + 1, (1,)))


def log_uniform(low, high):
    return math.exp(float(torch.empty(1).uniform_(math.log(low), math.log(high))))


def standardise(h):
    """Zero mean, unit variance per column, so the noise level is comparable."""
    return (h - h.mean(0)) / (h.std(0) + 1e-6)


def random_inputs(n_rows, n_columns, normal):
    """Input values: standard normal, or uniform with the same variance."""
    if normal:
        return torch.randn(n_rows, n_columns)
    return (torch.rand(n_rows, n_columns) * 2 - 1) * math.sqrt(3)


def sample_table(n_rows, d, prior):
    """One synthetic table: x (n_rows, d), y (n_rows,) with classes {0, 1}, and
    used (d,): True for the columns that are inputs of the MLP (for the app)."""
    # 1) How many of the d columns influence the target.
    k = d if prior == "all" else randint(1, d)

    # 2) Table-level settings.
    depth = randint(0, 3)
    width = randint(4, 16)
    weight_scale = log_uniform(0.5, 3.0)   # larger weights -> more wiggly functions
    noise_std = log_uniform(0.01, 0.5)     # larger noise -> less predictable labels
    normal = randint(0, 1) == 1            # same input distribution for all columns

    # 3) The k causes go through a random MLP, with noise at every node.
    causes = random_inputs(n_rows, k, normal)
    h = causes
    for _ in range(depth):
        activation = ACTIVATIONS[randint(0, len(ACTIVATIONS) - 1)]
        w = torch.randn(h.shape[1], width) * weight_scale / math.sqrt(h.shape[1])
        b = torch.randn(width)
        h = standardise(activation(h @ w + b)) + noise_std * torch.randn(n_rows, width)

    # 4) Target node: random linear read-out plus noise, split at the median.
    z = standardise(h @ torch.randn(h.shape[1], 1)).squeeze(1)
    z = z + noise_std * torch.randn(n_rows)
    y = (z > z.median()).long()

    # 5) The d - k other columns: same distribution, never used for y.
    #    (Under prior "all", k = d and this adds nothing.)
    unrelated = random_inputs(n_rows, d - k, normal)
    x = torch.cat([causes, unrelated], dim=1)
    order = torch.randperm(d)              # shuffle the column order
    return x[:, order], y, order < k       # the first k columns before shuffling were the causes


def sample_batch(batch_size, prior, max_columns=50, columns="log-uniform"):
    """A batch of tables that all have the same shape, so no padding is needed.
    The number of columns d lies between 2 and max_columns:
      "log-uniform"  every order of magnitude equally often, so small tables stay
                     frequent (used: 2 to 50; about 46 % have at most 8 columns),
      "uniform"      every d equally often (only in the stopped pilot run).
    Returns x (B, n_train + N_HELD_OUT, d), y (B, n_train + N_HELD_OUT), n_train."""
    if columns == "uniform":
        d = randint(2, max_columns)
    else:   # floor of a log-uniform number in [2, max_columns + 1)
        d = min(int(log_uniform(2, max_columns + 1)), max_columns)
    n_train = randint(16, 64)
    tables = [sample_table(n_train + N_HELD_OUT, d, prior) for _ in range(batch_size)]
    x = torch.stack([x for x, _, _ in tables])
    y = torch.stack([y for _, y, _ in tables])
    return x, y, n_train
