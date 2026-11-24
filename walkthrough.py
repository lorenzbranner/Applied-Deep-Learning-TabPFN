"""The whole way from prior to prediction, for the tab "Walkthrough" of the app.

trace() repeats MiniPFN.forward (model.py) line by line with the model's own layers,
but keeps every intermediate result and the attention weights. check_walkthrough.py
tests that it gives exactly the same probability as model.predict_proba.

Everything here uses moons: two columns, so every step can be drawn in the plane.
"""
import torch  # keep first: on Windows torch must load before scikit-learn

import json
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.datasets import make_moons

import tasks
from prior import ACTIVATIONS, N_HELD_OUT, log_uniform, randint, random_inputs, standardise
from style import BLUE, GREY, INK, RED, RULE

NAMES = {torch.tanh: "tanh", torch.relu: "relu", torch.sin: "sin", torch.abs: "abs",
         torch.square: "square", torch.sign: "sign"}
ACTIVATION_NAMES = [NAMES[a] for a in ACTIVATIONS]
NEGATIVE = "#a7aaa3"          # edge colour of a negative weight in the MLP drawing
TOKEN = LinearSegmentedColormap.from_list("token", ["#8f928c", "#ffffff", INK])  # -, 0, +

X1_RANGE, X2_RANGE = (-1.6, 2.6), (-1.1, 1.6)
# Light ends, so that the training rows (full blue and red) stay visible on the surface.
SURFACE = LinearSegmentedColormap.from_list("surface", ["#b4c6e2", "#ffffff", "#e2b9b8"])


def walk_run():
    """The pre-training run the app shows: run 1 of 3 (checkpoints/seed1)."""
    return tasks.CHECKPOINTS / "seed1"


def prior_settings(folder):
    """How the run drew its tables, read from its last checkpoint of prior B."""
    step = tasks.available_steps(folder, "some")[-1]
    checkpoint = torch.load(folder / f"some_{step:06d}.pt", map_location="cpu")
    return {"max_columns": checkpoint["max_columns"],
            "columns": checkpoint["columns"], "steps": step}


# --------------------------------------------------------------- the prior --
def draw_table(prior, seed, max_columns=50, columns="log-uniform", choose=None):
    """One table as pre-training draws it: prior.sample_batch(1, ...) line by line,
    with the same random numbers in the same order, but every random choice is kept.

    choose: values set by the sliders of the app, e.g. {"d": 12, "depth": 2}. The
    random number is still drawn and then replaced, so all later draws stay the same.
    check_walkthrough.py tests: without choose (or with choose = the drawn values),
    x, y and the causes are identical to prior.sample_batch after torch.manual_seed."""
    choose = choose or {}
    torch.manual_seed(seed)
    if columns == "uniform":                                   # 0) shape (once per batch)
        d = randint(2, max_columns)
    else:
        d = min(int(log_uniform(2, max_columns + 1)), max_columns)
    d = choose.get("d", d)
    n_train = choose.get("n_train", randint(16, 64))
    n_rows = n_train + N_HELD_OUT
    if prior == "all":                                         # 1) number of causes
        k = d
    else:
        k = randint(1, d)
        k = d - choose["unrelated"] if "unrelated" in choose else k
    depth = choose.get("depth", randint(0, 3))                 # 2) table-level settings
    width = choose.get("width", randint(4, 16))
    weight_scale = choose.get("weight_scale", log_uniform(0.5, 3.0))
    noise_std = choose.get("noise_std", log_uniform(0.01, 0.5))
    normal = choose.get("normal", randint(0, 1) == 1)
    causes = random_inputs(n_rows, k, normal)                  # 3) the random MLP
    h, layers = causes, []
    for i in range(depth):
        activation = ACTIVATIONS[randint(0, len(ACTIVATIONS) - 1)]
        if "activations" in choose:
            activation = ACTIVATIONS[ACTIVATION_NAMES.index(choose["activations"][i])]
        w = torch.randn(h.shape[1], width) * weight_scale / math.sqrt(h.shape[1])
        b = torch.randn(width)
        h = standardise(activation(h @ w + b)) + noise_std * torch.randn(n_rows, width)
        layers.append({"activation": NAMES[activation], "w": w.numpy(), "b": b.numpy()})
    readout = torch.randn(h.shape[1], 1)                       # 4) target node
    z = standardise(h @ readout).squeeze(1)
    z = z + noise_std * torch.randn(n_rows)
    y = (z > z.median()).long()
    unrelated = random_inputs(n_rows, d - k, normal)           # 5) unrelated columns
    x = torch.cat([causes, unrelated], dim=1)
    order = torch.randperm(d)
    return {"x": x[:, order].numpy(), "y": y.numpy(), "used": (order < k).numpy(),
            "order": order.numpy(), "k": k, "d": d, "n_train": n_train, "depth": depth, "width": width,
            "weight_scale": weight_scale, "noise_std": noise_std, "normal": normal,
            "layers": layers, "readout": readout.numpy()[:, 0], "z": z.numpy()}


def mlp_figure(t):
    """The drawn MLP: the columns of the table on the left (in their final, shuffled
    order), the hidden layers, the target node on the right. Edge width and darkness
    = size of the weight; dark = positive, light = negative."""
    d, k = t["d"], t["k"]
    n_layers = len(t["layers"])
    fig, ax = plt.subplots(figsize=(10, 3.6), layout="constrained")
    xs = np.arange(n_layers + 2)                                 # input, hidden..., target

    def ys(n):
        return np.linspace(1, -1, n) * min(1, n / 8)

    # Input nodes in the order of the table; column j holds original column order[j].
    in_y = ys(d)
    cause_y = {int(o): in_y[j] for j, o in enumerate(t["order"]) if o < k}  # original index -> y
    previous = [cause_y[i] for i in range(k)]                   # the k causes, in MLP order
    weights = [layer["w"] for layer in t["layers"]] + [t["readout"][:, None]]
    heights = [ys(len(layer["b"])) for layer in t["layers"]] + [np.array([0.0])]
    for column, (w, target_y) in enumerate(zip(weights, heights), start=1):
        biggest = np.abs(w).max()
        for i, y0 in enumerate(previous):
            for j, y1 in enumerate(target_y):
                share = abs(w[i, j]) / biggest
                ax.plot([xs[column - 1], xs[column]], [y0, y1], lw=0.3 + 2.6 * share,
                        color=INK if w[i, j] > 0 else NEGATIVE, alpha=0.15 + 0.85 * share,
                        zorder=1, solid_capstyle="round")
        previous = list(target_y)
    # Up to 50 columns: the nodes shrink with their spacing (about 87 pt per unit
    # of y); below 13 pt spacing only the first and the last column get a name.
    gap = 87 * (in_y[0] - in_y[1]) if d > 1 else 87
    size, named = min(16, 0.8 * gap) ** 2, gap >= 13
    for j in range(d):
        cause = bool(t["used"][j])
        ax.scatter(0, in_y[j], s=size, color=INK if cause else "white",
                   edgecolors=INK if cause else GREY, linewidths=1.2 if named else 0.6, zorder=3)
        if named or j in (0, d - 1):
            ax.text(-0.12, in_y[j], f"x{j + 1}" + ("  unrelated" if named and not cause else ""),
                    ha="right", va="center", color=INK if cause else GREY, fontsize=11)
    if not named:
        ax.text(-0.12, 0, f"{k} causes (filled)\n{d - k} unrelated (open)", ha="right",
                va="center", fontsize=11, color=GREY)
    for column, layer in enumerate(t["layers"], start=1):
        for y0 in heights[column - 1]:
            ax.scatter(column, y0, s=110, color="white", edgecolors=INK, linewidths=1, zorder=3)
        ax.text(column, 1.22, f"layer {column}\n{layer['activation']}", ha="center",
                va="bottom", fontsize=11, fontweight="bold")
    ax.scatter(xs[-1], 0, s=420, color="white", edgecolors=INK, linewidths=1.6, zorder=3)
    ax.text(xs[-1] + 0.12, 0, "target node z\n→ class 1 if z > median", va="center",
            fontsize=11)
    ax.text(0, 1.22, "columns", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set(xlim=(-0.9, xs[-1] + 1.1), ylim=(-1.15, 1.5))
    ax.axis("off")
    return fig


def target_figure(t):
    """How the classes come about: the target node z of all rows, split at its median."""
    fig, ax = plt.subplots(figsize=(3.6, 3.4), layout="constrained")
    z, median = t["z"], np.median(t["z"])
    width = (z.max() - z.min()) / 24            # equal bins, one edge exactly at the median
    bins = median + width * np.arange(np.floor((z.min() - median) / width),
                                      np.ceil((z.max() - median) / width) + 1)
    ax.hist(z[z <= median], bins=bins, color=BLUE)
    ax.hist(z[z > median], bins=bins, color=RED)
    ax.axvline(median, color=INK, lw=1.4)
    ax.text(median, ax.get_ylim()[1], " median", va="top", fontsize=10)
    ax.set(xlabel="target node z", yticks=[])
    ax.set_title("the label: z split at the median", loc="left")
    return fig


def features_figure(t, per_row=8):
    """Every column of the table as a histogram, mirrored: class 0 (blue) above the
    line, class 1 (red) below, the same bins for both. The two halves shifted against
    each other: the column separates the classes on its own; mirror images: it does
    not. Black title = cause, grey title = unrelated."""
    x, y, used, d = t["x"], t["y"], t["used"], t["d"]
    n_cols = min(d, per_row)
    n_rows = math.ceil(d / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8.4, 0.45 + 1.3 * n_rows),
                             layout="constrained", squeeze=False)
    for j, ax in enumerate(axes.flat):
        if j >= d:
            ax.axis("off")
            continue
        bins = np.linspace(x[:, j].min(), x[:, j].max(), 11)
        up, _ = np.histogram(x[y == 0, j], bins)
        down, _ = np.histogram(x[y == 1, j], bins)
        ax.stairs(up, bins, fill=True, color=BLUE)        # class 0 above the line
        ax.stairs(-down, bins, fill=True, color=RED)      # class 1 below it
        ax.axhline(0, color="white", lw=1)
        top = max(up.max(), down.max())
        ax.set_ylim(-top, top)
        cause = bool(used[j])
        ax.set_title(f"x{j + 1}" + (" cause" if cause else ""), fontsize=13,
                     color=INK if cause else GREY, fontweight="bold" if cause else "normal")
        ax.set(xticks=[], yticks=[])
        for side in ax.spines.values():
            side.set_visible(False)
    fig.suptitle(f"every column, all {len(y)} rows: class 0 up (blue), class 1 down (red)",
                 x=0.01, ha="left", fontsize=15, fontweight="bold")
    return fig


# ------------------------------------------------------------------- moons --
def moons(seed=0, n_train=50):
    """The training rows of moons that the walkthrough uses: 1,050 rows with two
    columns, shuffled with the seed, then the first n_train rows."""
    x, y = make_moons(1050, noise=0.2, random_state=0)
    order = np.random.RandomState(seed).permutation(len(y))[:n_train]
    return x[order], y[order]


@torch.no_grad()
def trace(model, x_train, y_train, x_query):
    """One forward pass for the rows x_query, step by step. Returns a dict:
    the values after every step for the first query row, and the attention weights."""
    x = torch.as_tensor(np.vstack([x_train, x_query]), dtype=torch.float32)[None]
    y_known = torch.as_tensor(y_train, dtype=torch.long)[None]
    B, R, F = x.shape
    n_train = y_known.shape[1]
    out = {"n_train": n_train, "raw": x[0, n_train].numpy()}

    # 1) z-normalise with the training rows (model.py, MiniPFN.forward)
    mean = x[:, :n_train].mean(1, keepdim=True)
    std = x[:, :n_train].std(1, keepdim=True) + 1e-6
    x = (x - mean) / std
    out.update(mean=mean[0, 0].numpy(), std=std[0, 0].numpy(), z=x[0, n_train].numpy(),
               z_train=x[0, :n_train].numpy())

    # 2) the target column: label for training rows, "unknown" flag for the query rows
    y = torch.zeros(B, R, 2)
    y[:, :n_train, 0] = y_known.float() * 2 - 1
    y[:, n_train:, 1] = 1.0

    # 3) one token per cell
    h = torch.cat([model.x_encoder(x.unsqueeze(-1)), model.y_encoder(y).unsqueeze(2)], dim=2)
    out["tokens_shape"] = tuple(h.shape)
    out["tokens"] = h[0, n_train].numpy()          # the star's 3 tokens (x1, x2, target)
    out["layers"] = []

    # 4) the layers (model.py, TwoWayLayer.forward), with the attention weights kept
    for layer in model.layers:
        C, D = h.shape[2], h.shape[3]
        s = h.reshape(B * R, C, D)
        update, feature_w = layer.feature_attention(s, s, s, need_weights=True)
        s = layer.norm1(s + update)
        h = s.reshape(B, R, C, D)
        after_feature = h[0, n_train].numpy()

        s = h.transpose(1, 2).reshape(B * C, R, D)
        train = s[:, :n_train]
        update, sample_w = layer.sample_attention(s, train, train, need_weights=True)
        s = layer.norm2(s + update)
        h = s.reshape(B, C, R, D).transpose(1, 2)
        after_sample = h[0, n_train].numpy()
        h = layer.norm3(h + layer.mlp(h))
        out["layers"].append({
            "after_feature": after_feature,                  # the star's 3 tokens after
            "after_sample": after_sample,                    # each part of the layer
            # how much the target token of the first query row reads x1, x2, itself
            "feature": feature_w[n_train, -1].numpy(),
            "feature_matrix": feature_w[n_train].numpy(),   # 3 x 3 for the star's row
            "tokens": h[0, n_train].numpy(),                 # the star's tokens after the layer
            # how much the target token of every query row reads each training row
            "sample": sample_w.reshape(B, C, R, n_train)[0, -1, n_train:].numpy(),
            "shape": tuple(h.shape)})

    # 5) decoder on the target token of the query rows, then softmax
    logits = model.decoder(h[:, n_train:, -1])
    out["token"] = h[0, n_train, -1].numpy()
    out["logits"] = logits[0, 0].numpy()
    out["proba"] = logits.softmax(-1)[0, :, 1].numpy()
    return out


def grid(n=70):
    g1, g2 = np.meshgrid(np.linspace(*X1_RANGE, n), np.linspace(*X2_RANGE, n))
    return g1, g2, np.column_stack([g1.ravel(), g2.ravel()])


def surface(model, x_train, y_train, n=70):
    """P(class 1) for every point of a grid, in ONE forward pass: the grid points are
    test rows, and test rows never read each other, so this equals n * n single passes."""
    g1, g2, points = grid(n)
    return g1, g2, model.predict_proba(x_train, y_train, points).reshape(g1.shape)


def draw_points(ax, x_train, y_train, sizes=None, alpha=None):
    for label, colour in ((0, BLUE), (1, RED)):
        keep = y_train == label
        ax.scatter(x_train[keep, 0], x_train[keep, 1], color=colour, edgecolors="white",
                   linewidths=0.8, zorder=3,
                   s=40 if sizes is None else sizes[keep],
                   alpha=1 if alpha is None else alpha[keep])


def draw_query(ax, point):
    ax.scatter(*point, marker="*", s=380, color=INK, edgecolors="white", linewidths=1.2,
               zorder=5)


def style_plane(ax):
    ax.set(xlim=X1_RANGE, ylim=X2_RANGE, xticks=[], yticks=[], aspect="equal")
    for side in ax.spines.values():
        side.set_visible(True)
        side.set(color=RULE, linewidth=0.8)


def training_figure(history_a, history_b, surfaces, x_train, y_train):
    """Left: loss of both priors during pre-training. Right: what mini-PFN B
    predicts on the same 50 moons rows after 0, 100, ... steps."""
    fig = plt.figure(figsize=(11, 2.9), layout="constrained")
    left, right = fig.subfigures(1, 2, width_ratios=[1, 2.6])
    ax = left.subplots()
    for history, colour, name in ((history_a, "#9a9da1", "A"), (history_b, INK, "B")):
        steps = [h["step"] for h in history]
        ax.plot(steps, [h["loss"] for h in history], color=colour, lw=2, label=f"prior {name}")
    ax.axhline(np.log(2), color=GREY, lw=0.8, ls=":")
    ax.text(ax.get_xlim()[1], np.log(2), "guessing ", ha="right", va="bottom",
            color=GREY, fontsize=9)
    ax.set(xlabel="pre-training step", ylabel="loss on new\nsynthetic tables")
    ax.legend(loc="upper right")
    axes = right.subplots(1, len(surfaces))
    for ax, (step, (g1, g2, p)) in zip(axes, surfaces.items()):
        ax.contourf(g1, g2, p, levels=np.linspace(0, 1, 11), cmap=SURFACE)
        draw_points(ax, x_train, y_train)
        ax.set_title(f"{step:,} steps", fontsize=11)
        style_plane(ax)
    return fig


def load_history(folder, prior):
    return json.loads((folder / f"{prior}_history.json").read_text())["history"]


# ------------------------------------------- the forward pass, step by step --
STEPS = ["Input", "z-normalise", "Embedding",
         "Layer 1 · feature attention", "Layer 1 · sample attention", "Layer 1 · MLP",
         "Layer 2", "Layer 3", "Decoder and softmax", "Output"]

STEP_TEXT = {
    "Input": "The table is the input: training rows with label, the star without.",
    "z-normalise": "Each column is shifted and scaled with the training rows only.",
    "Embedding": "One token of 64 numbers per cell. The star's target cell says “unknown”.",
    "Layer 1 · feature attention": "Within the row: the target cell collects the star's "
                                   "features. Then residual + LayerNorm.",
    "Layer 1 · sample attention": "Within the column, keys = training rows only: the star "
                                  "never reads other test rows. Then residual + LayerNorm.",
    "Layer 1 · MLP": "The same small MLP on every token. Then residual + LayerNorm.",
    "Layer 2": "Same three parts as layer 1, with its own weights. Shown: the result.",
    "Layer 3": "Same three parts as layer 1, with its own weights. Shown: the result.",
    "Decoder and softmax": "Only the target cell of the star is read out.",
    "Output": "Test rows never read each other, so the whole plane takes one pass. "
              "Black line: P = 0.5.",
}


def step_text(step):
    return STEP_TEXT[STEPS[step]]


# The chain of blocks above the big picture: (name, shape, parts); a step lights up
# one block, or one part of the first layer.
CHAIN = [("Input", "(51, 2)", 1), ("z-normalise", "(51, 2)", 1), ("Embedding", "(51, 3, 64)", 1),
         ("Layer 1", "(51, 3, 64)", 3), ("Layer 2", "(51, 3, 64)", 3),
         ("Layer 3", "(51, 3, 64)", 3), ("Decoder\n+ softmax", "(1, 2)", 1),
         ("Output", "P(class 1)", 1)]
STEP_IN_CHAIN = [(0, None), (1, None), (2, None), (3, 0), (3, 1), (3, 2), (4, None),
                 (5, None), (6, None), (7, None)]


def _box3d(ax, x, y, w, h, colour, edge):
    """A box with a visible top and right side: the front, then the two slanted faces."""
    dx, dy = 0.16, 0.14
    ax.add_patch(plt.Polygon([(x, y + h), (x + dx, y + h + dy), (x + w + dx, y + h + dy),
                              (x + w, y + h)], color=colour, ec=edge, lw=0.8))
    ax.add_patch(plt.Polygon([(x + w, y), (x + w + dx, y + dy), (x + w + dx, y + h + dy),
                              (x + w, y + h)], color=colour, ec=edge, lw=0.8))
    ax.add_patch(plt.Rectangle((x, y), w, h, color=colour, ec=edge, lw=0.8))


def chain_figure(step):
    """The whole model as a chain of 3D blocks; the current step in black."""
    block, part = STEP_IN_CHAIN[step]
    fig, ax = plt.subplots(figsize=(11, 2.2), layout="constrained")
    x = 0.0
    for i, (name, shape, parts) in enumerate(CHAIN):
        width = 0.55 if parts == 1 else 0.3
        done = i < block
        for p in range(parts):
            active = i == block and (part is None or p == part)
            colour = INK if active else ("#c9ccc6" if done or (i == block and p < (part or 0))
                                         else "#f0f1ee")
            height = 1.0 if parts == 1 else 1.2
            _box3d(ax, x + p * (width + 0.08), -height / 2, width, height, colour, GREY)
            if parts == 3:
                label = ("feat.", "samp.", "MLP")[p]
                ax.text(x + p * (width + 0.08) + width / 2, 0, label, rotation=90,
                        ha="center", va="center", fontsize=8,
                        color="white" if active else GREY)
        total = parts * width + (parts - 1) * 0.08
        ax.text(x + total / 2, -0.95, name, ha="center", va="top", fontsize=10,
                fontweight="bold" if i == block else "normal", color=INK)
        ax.text(x + total / 2, -1.45 if "\n" not in name else -1.75, shape, ha="center",
                va="top", fontsize=8, color=GREY)
        if i < len(CHAIN) - 1:
            ax.annotate("", xy=(x + total + 0.68, 0), xytext=(x + total + 0.26, 0),
                        arrowprops={"arrowstyle": "->", "color": GREY, "lw": 1})
        x += total + 0.75
    ax.set(xlim=(-0.2, x), ylim=(-2.0, 1.0))
    ax.set_aspect("equal")
    ax.axis("off")
    return fig


def _cells(ax, rows, colours, header):
    """A small table drawn as cells: rows of texts, one colour per cell (None = light)."""
    for i, (row, row_colours) in enumerate(zip(rows, colours)):
        for j, (text, colour) in enumerate(zip(row, row_colours)):
            ax.add_patch(plt.Rectangle((j, -i), 0.94, -0.9, color=colour or "#f0f1ee"))
            strong = colour in (BLUE, RED, INK)
            ax.text(j + 0.47, -i - 0.45, text, ha="center", va="center", fontsize=11,
                    color="white" if strong else INK, fontweight="bold" if strong else "normal")
    for j, text in enumerate(header):
        ax.text(j + 0.47, 0.25, text, ha="center", va="bottom", fontsize=11, color=GREY)
    ax.set(xlim=(-0.6, len(header) + 0.1), ylim=(-len(rows) - 0.2, 1))
    ax.axis("off")


def _strip(ax, values, label):
    """One token: its 64 numbers as a strip of cells (dark = large, light = small)."""
    limit = max(np.abs(values).max(), 1e-6)
    ax.imshow(np.atleast_2d(values), cmap=TOKEN, vmin=-limit, vmax=limit, aspect="auto")
    ax.set(xticks=[], yticks=[])
    ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=11, color=INK)


def _input(t, x_train, y_train, point):
    fig, (plane, table) = plt.subplots(1, 2, figsize=(10, 3.6), width_ratios=[1.2, 1],
                                       layout="constrained")
    draw_points(plane, x_train, y_train)
    draw_query(plane, point)
    style_plane(plane)
    plane.set_title("50 training rows and the star")
    rows = [[f"{a:.2f}", f"{b:.2f}", str(c)] for (a, b), c in zip(x_train[:5], y_train[:5])]
    colours = [[None, None, RED if c else BLUE] for c in y_train[:5]]
    rows += [["⋮"] * 3, [f"{point[0]:.2f}", f"{point[1]:.2f}", "?"]]
    colours += [["white"] * 3, [None, None, INK]]
    _cells(table, rows, colours, ["x1", "x2", "target"])
    table.set_title("what the network gets: one table, 51 rows")
    return fig


def _normalise(t, x_train, y_train, point):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
    for ax, xs, star, centre, title in (
            (axes[0], x_train, point, t["mean"], "before"),
            (axes[1], t["z_train"], t["z"], (0, 0), "after: (x − mean) / std of the training rows")):
        ax.axhline(centre[1], color=RULE, lw=1, zorder=0)
        ax.axvline(centre[0], color=RULE, lw=1, zorder=0)
        draw_points(ax, xs, y_train)
        draw_query(ax, star)
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="datalim")
    return fig


def _tokens(t):
    fig, axes = plt.subplots(3, 1, figsize=(10, 2.6), layout="constrained")
    labels = [f"x1 = {t['z'][0]:.2f}  →  x_encoder  →",
              f"x2 = {t['z'][1]:.2f}  →  x_encoder  →",
              "target = unknown  →  y_encoder  →"]
    for ax, values, label in zip(axes, t["tokens"], labels):
        _strip(ax, values, label)
    axes[0].set_title("every cell of the star's row becomes a token of 64 numbers")
    return fig


def _feature(layer):
    fig, (bars, tokens) = plt.subplots(1, 2, figsize=(10, 3.2), width_ratios=[1, 1.4],
                                       layout="constrained")
    share = layer["feature"]
    bars.barh([2, 1, 0], share, color=[INK, INK, GREY], height=0.6)
    for y0, s in zip([2, 1, 0], share):
        bars.text(s + 0.02, y0, f"{s:.0%}", va="center", fontsize=12)
    bars.set(yticks=[2, 1, 0], yticklabels=["x1", "x2", "target (itself)"],
             xlim=(0, 1.2), xticks=[])
    bars.set_title("the target cell of the star reads its row")
    _tokens3(tokens, layer["after_feature"], "the star's 3 tokens after feature attention")
    return fig


def _tokens3(ax, values, title):
    """The 3 tokens of the star's row (x1, x2, target) as 3 strips of 64 numbers."""
    limit = np.abs(values).max()
    ax.imshow(values, cmap=TOKEN, vmin=-limit, vmax=limit, aspect="auto")
    ax.set(xticks=[], yticks=[0, 1, 2], yticklabels=["x1", "x2", "target"])
    ax.set_title(title)


def _mlp(layer):
    """Layer 1, part 3: the MLP 64 -> 128 -> 64 on every token, + residual + LayerNorm."""
    fig, (before, after) = plt.subplots(1, 2, figsize=(10, 2.8), layout="constrained")
    _tokens3(before, layer["after_sample"], "before: after sample attention")
    _tokens3(after, layer["tokens"], "after: MLP 64 → 128 → 64, + residual, LayerNorm")
    return fig


def _block(layer, x_train, y_train, point, number):
    """Layers 2 and 3: only the result of the block, and what its sample attention read."""
    fig, (plane, tokens) = plt.subplots(1, 2, figsize=(10, 3.4), width_ratios=[1.2, 1],
                                        layout="constrained")
    weights = layer["sample"][0]
    share = np.clip(weights * len(weights) / 4, 0, 1)
    draw_points(plane, x_train, y_train, sizes=20 + 480 * share, alpha=0.3 + 0.7 * share)
    draw_query(plane, point)
    style_plane(plane)
    plane.set_title(f"rows read in layer {number} (largest {weights.max():.0%})")
    _tokens3(tokens, layer["tokens"], f"the star's 3 tokens after layer {number}")
    return fig


def _sample(layer, x_train, y_train, point):
    fig, ax = plt.subplots(figsize=(10, 3.8), layout="constrained")
    weights = layer["sample"][0]
    even = 1 / len(weights)                      # every row read equally: 2 % each
    share = np.clip(weights / even / 4, 0, 1)    # 4 x the even weight or more = largest
    draw_points(ax, x_train, y_train, sizes=20 + 480 * share, alpha=0.3 + 0.7 * share)
    draw_query(ax, point)
    style_plane(ax)
    ax.set_title(f"the target cell of the star reads the training rows of its column "
                 f"(evenly: {even:.0%} each; largest here: {weights.max():.0%})")
    return fig


def _decoder(t):
    fig = plt.figure(figsize=(10, 3.2), layout="constrained")
    token, logit, prob = fig.subplots(1, 3, width_ratios=[2.2, 1, 1])
    _strip(token, t["layers"][-1]["tokens"][2], "")
    token.set_title("target token of the star (64)\n→ MLP 64 → 128 → 2")
    l0, l1 = t["logits"]
    logit.bar([0, 1], [l0, l1], color=[BLUE, RED], width=0.6)
    logit.axhline(0, color=RULE, lw=1)
    logit.set(xticks=[0, 1], xticklabels=["class 0", "class 1"], yticks=[])
    logit.set_title("2 logits")
    tall = 0.3 * max(abs(l0), abs(l1), 1e-6)
    for xx, v in ((0, l0), (1, l1)):
        if abs(v) >= tall:                       # label inside a tall bar
            logit.text(xx, v / 2, f"{v:.2f}", ha="center", va="center", color="white",
                       fontweight="bold")
        else:                                    # next to a short one
            logit.text(xx, v, f"{v:.2f}", ha="center", va="bottom" if v >= 0 else "top",
                       fontweight="bold")
    p1 = float(t["proba"][0])
    prob.bar([0, 1], [1 - p1, p1], color=[BLUE, RED], width=0.6)
    prob.set(xticks=[0, 1], xticklabels=["class 0", "class 1"], yticks=[], ylim=(0, 1.15))
    prob.set_title("softmax")
    for xx, v in ((0, 1 - p1), (1, p1)):
        prob.text(xx, v + 0.02, f"{v:.0%}", ha="center", va="bottom", fontweight="bold")
    return fig


def _plane(surface, x_train, y_train, point):
    g1, g2, p = surface
    fig, ax = plt.subplots(figsize=(10, 3.8), layout="constrained")
    contour = ax.contourf(g1, g2, p, levels=np.linspace(0, 1, 11), cmap=SURFACE)
    ax.contour(g1, g2, p, levels=[0.5], colors=INK, linewidths=1.2)
    draw_points(ax, x_train, y_train)
    draw_query(ax, point)
    style_plane(ax)
    fig.colorbar(contour, ax=ax, ticks=[0, 0.5, 1], shrink=0.8, label="P(class 1)")
    ax.set_title("the same pass for every point of a 70 × 70 grid at once")
    return fig


def step_figure(step, t, x_train, y_train, point, surface):
    """The picture of one step of the forward pass (STEPS[step]) for the star."""
    name = STEPS[step]
    first = t["layers"][0]
    pictures = {
        "Input": lambda: _input(t, x_train, y_train, point),
        "z-normalise": lambda: _normalise(t, x_train, y_train, point),
        "Embedding": lambda: _tokens(t),
        "Layer 1 · feature attention": lambda: _feature(first),
        "Layer 1 · sample attention": lambda: _sample(first, x_train, y_train, point),
        "Layer 1 · MLP": lambda: _mlp(first),
        "Layer 2": lambda: _block(t["layers"][1], x_train, y_train, point, 2),
        "Layer 3": lambda: _block(t["layers"][2], x_train, y_train, point, 3),
        "Decoder and softmax": lambda: _decoder(t),
        "Output": lambda: _plane(surface, x_train, y_train, point),
    }
    return pictures[name]()
