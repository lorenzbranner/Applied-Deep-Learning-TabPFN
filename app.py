"""Live demo: does the prior teach the mini-PFN to ignore uninformative columns?

Run from this folder:   streamlit run app.py
Streamlit runs this script from top to bottom on every change of a control;
st.cache_data remembers results that were computed before. The look of the
page is set in .streamlit/config.toml, the look of the figures in style.py.
"""
import torch  # keep first: on Windows torch must load before scikit-learn

import io
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import experiment
import tasks
import walkthrough as wt
from style import COLOURS, GREY, INK, RED, RULE, frame, headline

RUN = wt.walk_run()                  # run 1 of the new prior (2 to 50 columns) once it exists
SEEDS = range(5)                     # the ROC AUC is the mean over 5 splits (= splits 0-4 of experiment.py)
A, B = tasks.PRIORS["all"], tasks.PRIORS["some"]
RESULTS = experiment.OUT                          # experiment.py writes here
SELECTION = {t: s for t, s in json.loads((experiment.DATA / "selection.json").read_text()).items()
             if s["passes_gate"]}           # the 12 tables and their chosen columns

st.set_page_config(page_title="Uninformative columns", layout="wide")
st.html("<style>.stMainBlockContainer {padding-top: 2.5rem}</style>")   # less empty space on top


def show(figure):
    st.pyplot(figure)
    plt.close(figure)


@st.cache_resource
def load_mini_pfn(prior, step):
    return tasks.mini_pfn(RUN, prior, step)


@st.cache_data(show_spinner=False)
def load_table(table):
    """The 12 tables: chosen columns, rows outside the selection part."""
    return experiment.load(table, SELECTION[table])


@st.cache_data(show_spinner=False)
def roc_aucs(name, table, n_added, n_train, step=None):
    """Test ROC AUC of one model for each split in SEEDS; the splits of experiment.py."""
    if name in (A, B):
        prior = "all" if name == A else "some"
        predict = load_mini_pfn(prior, step).predict_proba
    elif name == tasks.OFFICIAL:
        def predict(x_train, y_train, x_test):   # 1 ensemble member instead of 4: 7x faster
            return tasks.official_tabpfn(x_train, y_train, x_test, n_estimators=1)
    else:
        predict = tasks.BASELINES[name]
    result = []
    for seed in SEEDS:
        x, y = load_table(table)
        x_train, y_train, x_test, y_test = experiment.make_task(x, y, n_added, seed, n_train)
        result.append(experiment.roc_auc(predict(x_train, y_train, x_test), y_test))
    return result


def bars(results):
    """One small panel per model: the mean ROC AUC as a bar and in large type,
    the single splits as dots."""
    fig, axes = plt.subplots(1, len(results), figsize=(1.6 * len(results), 3.3), sharey=True)
    for ax, (name, values) in zip(axes, results.items()):
        ax.bar(0, np.mean(values), width=0.7, color=COLOURS[name])
        ax.scatter(np.linspace(-0.2, 0.2, len(values)), values, s=26, color="white",
                   edgecolors=INK, linewidths=1, zorder=3)
        ax.axhline(0.5, color=GREY, lw=0.8, ls=":")          # guessing on two classes
        headline(ax, name, f"{np.mean(values):.3f}")
        ax.set(xlim=(-0.6, 0.6), ylim=(0.4, 1.0), xticks=[])
        frame(ax, name)
    axes[0].set_ylabel("test ROC AUC")
    fig.subplots_adjust(top=0.70, wspace=0.25)
    return fig


@st.cache_data
def precomputed(changed):
    """Precomputed experiment: one ROC AUC per table, model and number of added
    columns (mean over 10 splits and, for A and B, over 3 pre-training runs).
    changed = modification time of auc.csv, so that new results replace old ones."""
    runs = pd.read_csv(RESULTS / "auc.csv")
    per_table = runs.groupby(["table", "model", "n_added"]).roc_auc.mean().reset_index()
    tests = pd.read_csv(RESULTS / "wilcoxon.csv") if (RESULTS / "wilcoxon.csv").exists() else None
    return per_table, tests


@st.cache_data(show_spinner=False)
def experiment_png(kind, changed):
    """The two figures of the Experiments tab as images: drawn once per results file,
    not on every click elsewhere on the page."""
    per_table, _ = precomputed(changed)
    names = [A, B, tasks.OFFICIAL] + list(tasks.BASELINES)
    fig = (mean_curve if kind == "mean" else table_curves)(per_table, names)
    image = io.BytesIO()
    fig.savefig(image, format="png")
    plt.close(fig)
    return image.getvalue()


def line(ax, x, y, name):
    ax.plot(x, y, label=name, color=COLOURS[name], lw=2.6 if name == B else 1.6,
            marker="o", ms=5 if name == B else 4, mec="white", zorder=3 if name == B else 2)


def mean_curve(per_table, names):
    """Mean ROC AUC over the tables against the number of added columns."""
    fig, ax = plt.subplots(figsize=(9, 3.2), layout="constrained")
    for name in names:
        mean = per_table[per_table.model == name].groupby("n_added").roc_auc.mean()
        line(ax, mean.index, mean.values, name)
    ax.set(xlabel="added uninformative columns", ylabel="ROC AUC, mean over the tables",
           xticks=sorted(per_table.n_added.unique()))
    ax.grid(axis="y", color=RULE, lw=0.6)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    return fig


def table_curves(per_table, names):
    """One small panel per table; each panel has its own y scale."""
    tables = sorted(per_table.table.unique(), key=str.lower)
    cols = 4
    rows = -(-len(tables) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 2.5 * rows), sharex=True,
                             layout="constrained", squeeze=False)
    for ax, table in zip(axes.flat, tables):
        part = per_table[per_table.table == table]
        for name in names:
            values = part[part.model == name].sort_values("n_added")
            line(ax, values.n_added, values.roc_auc, name)
        ax.set_title(table if len(table) < 26 else table[:24] + "…", fontsize=11)
        ax.set_xticks(sorted(per_table.n_added.unique()))
        ax.grid(axis="y", color=RULE, lw=0.6)
    for ax in axes.flat[len(tables):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("added columns")
    for ax in axes[:, 0]:
        ax.set_ylabel("ROC AUC")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside upper center",
               ncol=len(names), fontsize=10)
    return fig


# ------------------------------------------------------------------- page --
st.markdown("### Can a prior teach a network to ignore uninformative columns?")
st.caption("Demo for Hollmann et al., *Accurate predictions on small data with a tabular "
           "foundation model*, Nature 637 (2025). Mini-PFN A was pre-trained on synthetic "
           "tables in which every column influences the target, mini-PFN B on tables in "
           "which some columns do not. Nothing else differs.")

with st.sidebar:
    table = st.selectbox("Table", list(SELECTION), index=list(SELECTION).index("qsar-biodeg"))
    n_added = st.slider("Added uninformative columns", 0, 30, 0)
    n_train = st.slider("Training rows", 10, 200, 50, step=10)

step = tasks.available_steps(RUN, "some")[-1]   # fully pre-trained (learning: see Walkthrough)

live, experiments_tab, walk_tab = st.tabs(["Live", "Experiments", "Walkthrough"])

with live:
    names = [A, B, tasks.OFFICIAL] + list(tasks.BASELINES)   # TabPFN v2 always runs
    with st.spinner("Running the models ..."):
        results = {name: roc_aucs(name, table, n_added, n_train, step if name in (A, B) else None)
                   for name in names}
    show(bars(results))
    st.caption(f"Bar and large number: mean test ROC AUC over {len(SEEDS)} random splits "
               f"(dots), the first {len(SEEDS)} splits of the experiment: stratified training "
               f"rows, at most 500 test rows; every split has its own added columns. "
               f"Dotted line: 0.5, guessing. Mini-PFN A and B: pre-training run 1 of 3. "
               f"TabPFN v2 runs here with 1 ensemble member instead of 4, so that it answers "
               f"in seconds.")

# ------------------------------------------------------------- experiments --
with experiments_tab:
    st.caption("Precomputed with experiment.py as written down beforehand in "
               "PREREGISTRATION.md. The sidebar does not change this tab.")
    if not (RESULTS / "auc.csv").exists():
        st.info("No results yet: run  python experiment.py --steps 3000")
    else:
        changed = (RESULTS / "auc.csv").stat().st_mtime
        per_table, tests = precomputed(changed)
        n_tables = per_table.table.nunique()
        st.markdown(f"#### {n_tables} real tables of the paper's benchmark, "
                    f"0 to 30 added uninformative columns")
        if tests is None:
            st.warning("The evaluation is still running: the tables so far, without the test.")
        else:
            main = tests[tests.n_added == tests.n_added.max()].iloc[0]
            start = tests[tests.n_added == 0].iloc[0]
            verdict = "significant" if main["wilcoxon p"] < 0.05 else "not significant"
            st.markdown(
                f"**Main test** (30 added columns): mini-PFN B has the higher ROC AUC on "
                f"**{int(main['B wins'])} of {int(main['tables'])}** tables; mean "
                f"A **{main['mean A']:.3f}**, B **{main['mean B']:.3f}**; two-sided Wilcoxon "
                f"p = **{main['wilcoxon p']:.4f}** ({verdict} at α = 0.05).  \n"
                f"But B is ahead already without added columns ({int(start['B wins'])} of "
                f"{int(start['tables'])}, p = {start['wilcoxon p']:.4f}): the gap grows from "
                f"{start['mean B'] - start['mean A']:.3f} to "
                f"{main['mean B'] - main['mean A']:.3f}.")
        st.image(experiment_png("mean", changed), use_container_width=True)
        st.caption("Each table counts once: its ROC AUC is the mean over 10 random splits "
                   "and, for mini-PFN A and B, over 3 pre-training runs (prior with 2 to 50 "
                   "columns, 3,000 steps). Every split: 50 stratified training rows, at most "
                   "500 test rows, columns chosen on a separate 25 % of the rows. TabPFN v2 "
                   "runs with its default 4 ensemble members.")
        st.image(experiment_png("tables", changed), use_container_width=True)
        st.caption("Every table on its own. Note: each panel has its own y scale.")
        if tests is not None:
            with st.expander("The test for every number of added columns"):
                st.table(tests.set_index("n_added").rename_axis("added columns")
                         .style.format({"mean A": "{:.3f}", "mean B": "{:.3f}",
                                        "wilcoxon p": "{:.4f}"}))

# ------------------------------------------------------------- walkthrough --
WALK = RUN                           # the same pre-training run as the Live tab
WALK_PRIOR = wt.prior_settings(WALK)
for key, start in (("walk_split", 0), ("walk_step", 0)):
    st.session_state.setdefault(key, start)


def bump(key):
    """Button: draw something new (a new seed)."""
    st.session_state[key] += 1


def go(delta):
    """Buttons Back / Next of the forward pass."""
    st.session_state.walk_step = min(max(st.session_state.walk_step + delta, 0),
                                     len(wt.STEPS) - 1)


@st.cache_resource
def load_walk_model(step):
    return tasks.mini_pfn(WALK, "some", step)


@st.cache_data(show_spinner=False)
def walk_surfaces(split):
    """P(class 1) over the moons plane after every saved pre-training step of B."""
    x_train, y_train = wt.moons(split)
    return {step: wt.surface(load_walk_model(step), x_train, y_train)
            for step in tasks.available_steps(WALK, "some")}


def log_stops(low, high, n=37):
    """Slider stops for a log-uniform choice: equal steps on the log scale."""
    return [float(f"{v:.3g}") for v in np.geomspace(low, high, n)]


def snap(value, stops):
    """The stop nearest to a drawn value (on the log scale)."""
    return min(stops, key=lambda stop: abs(np.log(stop) - np.log(value)))


WEIGHT_STOPS, NOISE_STOPS = log_stops(0.5, 3.0), log_stops(0.01, 0.5)


def new_table():
    """Button "Draw a new table": the next seed, drawn exactly as in pre-training;
    every slider jumps to the value that was drawn."""
    st.session_state.walk_table += 1
    t = wt.draw_table(st.session_state.walk_prior, st.session_state.walk_table,
                      WALK_PRIOR["max_columns"], WALK_PRIOR["columns"])
    st.session_state.walk.update(
        d=t["d"], unrelated=t["d"] - t["k"], n_train=t["n_train"], depth=t["depth"],
        width=t["width"], weight_scale=snap(t["weight_scale"], WEIGHT_STOPS),
        noise_std=snap(t["noise_std"], NOISE_STOPS),
        inputs="normal" if t["normal"] else "uniform",
        **{f"act{i + 1}": layer["activation"] for i, layer in enumerate(t["layers"])})
    st.session_state.walk_exact = {"weight_scale": t["weight_scale"],
                                   "noise_std": t["noise_std"]}


def remember(name, key):
    st.session_state.walk[name] = st.session_state[key]


def setting(column, name, label, options, help=None, disabled=False, fmt=str, key=None):
    """One slider per random choice of the prior. Its value is kept in
    st.session_state.walk, so that "Draw a new table" can set it."""
    key = key or "w_" + name
    st.session_state[key] = 0 if key == "w_unrelated_a" else st.session_state.walk[name]
    column.select_slider(label, options, key=key, help=help, disabled=disabled,
                         format_func=fmt, on_change=remember, args=(name, key))


if "walk" not in st.session_state:           # the first table: seed 1
    st.session_state.walk_table, st.session_state.walk_prior = 0, "some"
    st.session_state.walk = {"act1": "tanh", "act2": "tanh", "act3": "tanh"}
    new_table()


with walk_tab:
    st.caption(f"From the prior to one prediction, with mini-PFN B (pre-training run 1, "
               f"{WALK.parent.name}/{WALK.name}). The sidebar does not change this tab. "
               f"Every number is computed now (walkthrough.py, checked by check_walkthrough.py).")

    st.markdown("#### 1 The prior: how one synthetic table is drawn")
    st.html(f"""<style>      /* prior A has no unrelated columns: its slider is red and fixed */
      .st-key-w_unrelated_a [role="slider"] {{background-color: {RED}; border-color: {RED}}}
      .st-key-w_unrelated_a [data-testid="stSliderThumbValue"],
      .st-key-w_unrelated_a [data-testid="stWidgetLabel"] p {{color: {RED}}}
    </style>""")
    controls = st.columns([1.2, 3.4, 1.4])
    prior_choice = controls[0].radio("Prior", ["some", "all"], horizontal=True, key="walk_prior",
                                     format_func={"some": "B", "all": "A"}.get)
    controls[1].caption("Every slider is one random choice of the prior; its ends are the "
                        "values pre-training can draw. The button draws a new table exactly as "
                        "pre-training does and sets every slider to the value drawn.")
    controls[2].button("Draw a new table", on_click=new_table, use_container_width=True)
    walk = st.session_state.walk
    walk["unrelated"] = min(walk["unrelated"], walk["d"] - 1)
    rule = "log-uniform" if WALK_PRIOR["columns"] != "uniform" else "uniform"
    shape = st.columns(3)
    setting(shape[0], "d", f"Columns · {rule}",
            list(range(2, WALK_PRIOR["max_columns"] + 1)),
            help="Drawn once per pre-training step, for all 32 tables of the step. "
                 "Log-uniform: from 2 to 5 as often as from 5 to 12 or from 12 to 30.")
    if prior_choice == "some":
        setting(shape[1], "unrelated", "Unrelated columns · uniform",
                list(range(walk["d"])),
                help="The number of causes k is drawn from 1 to the number of columns; "
                     "the other columns never enter the MLP.")
    else:
        setting(shape[1], "unrelated", "Unrelated columns · prior A: always 0",
                list(range(walk["d"])), key="w_unrelated_a", disabled=True,
                help="Prior A: every column is a cause (k = number of columns).")
    setting(shape[2], "n_train", "Rows with label · uniform", list(range(16, 65)),
            help="Drawn once per pre-training step. Each table has 32 more rows whose "
                 "label the model must predict.")
    mlp = st.columns(4)
    setting(mlp[0], "depth", "Hidden layers · uniform", [0, 1, 2, 3])
    setting(mlp[1], "width", "Nodes per layer · uniform", list(range(4, 17)))
    setting(mlp[2], "weight_scale", "Weight scale · log-uniform", WEIGHT_STOPS,
            fmt="{:.2f}".format, help="Larger weights: a more wiggly function.")
    setting(mlp[3], "noise_std", "Noise std · log-uniform", NOISE_STOPS,
            fmt="{:.3f}".format, help="Larger noise: labels that are harder to predict.")
    nodes = st.columns(4)
    setting(nodes[0], "inputs", "Inputs · 50 : 50", ["uniform", "normal"],
            help="The same distribution for every column; uniform with variance 1.")
    for i in (1, 2, 3):
        setting(nodes[i], f"act{i}", f"Activation, layer {i} · uniform",
                wt.ACTIVATION_NAMES, disabled=i > walk["depth"])

    choose = {name: walk[name] for name in ("d", "unrelated", "n_train", "depth", "width")}
    choose["normal"] = walk["inputs"] == "normal"
    choose["activations"] = [walk[f"act{i}"] for i in (1, 2, 3)]
    for name in ("weight_scale", "noise_std"):   # slider not moved: the exact drawn value
        exact = st.session_state.walk_exact[name]
        stops = WEIGHT_STOPS if name == "weight_scale" else NOISE_STOPS
        choose[name] = exact if snap(exact, stops) == walk[name] else walk[name]
    drawn = wt.draw_table(prior_choice, st.session_state.walk_table, WALK_PRIOR["max_columns"],
                          WALK_PRIOR["columns"], choose)
    st.markdown(
        "Drawn from the seed, without a slider: the weights N(0, 1) × weight scale / "
        "√(inputs of the layer) and the biases N(0, 1); the input values. Every node: "
        "standardise(activation(inputs · weights + bias)) + noise. Target z = random "
        "read-out of the last layer + noise, class 1 if z > median. The unrelated columns "
        "are drawn like the inputs and never enter the MLP; then the column order is shuffled.")
    show(wt.mlp_figure(drawn))
    sides = st.columns([1, 2.4])
    with sides[0]:
        show(wt.target_figure(drawn))
    with sides[1]:
        show(wt.features_figure(drawn))
    st.caption(f"{drawn['n_train']} rows with label and {len(drawn['y']) - drawn['n_train']} "
               f"whose label the model must predict. Grey title: unrelated column. A cause need not separate the classes on "
               f"its own: the MLP mixes the causes, so the classes may show only in a "
               f"combination of columns. Pre-training draws 32 such tables per step: columns "
               f"and rows once for the step, everything else per table.")

    st.markdown("#### 2 Pre-training: the only time the weights change")
    split = st.session_state.walk_split
    x_train, y_train = wt.moons(split)
    show(wt.training_figure(wt.load_history(WALK, "all"), wt.load_history(WALK, "some"),
                            walk_surfaces(split), x_train, y_train))
    st.caption(f"Left: loss on new synthetic tables (mean of 100 steps; dotted: guessing). "
               f"Right: mini-PFN B on the same 50 moons rows after 0 to "
               f"{WALK_PRIOR['steps']:,} steps. Same rows, different weights.")

    st.markdown("#### 3 The forward pass, step by step")
    controls = st.columns([2, 2, 1.4])
    x1 = controls[0].slider("x1 of the star", -1.5, 2.5, 0.5, 0.05)
    x2 = controls[1].slider("x2 of the star", -1.0, 1.5, 0.25, 0.05)
    controls[2].button("Draw new training rows", on_click=bump, args=("walk_split",),
                       use_container_width=True)
    t = wt.trace(load_walk_model(WALK_PRIOR["steps"]), x_train, y_train, np.array([[x1, x2]]))
    step = st.session_state.walk_step
    nav = st.columns([1, 1, 6])
    nav[0].button("◀ Back", on_click=go, args=(-1,), disabled=step == 0,
                  use_container_width=True)
    nav[1].button("Next ▶", on_click=go, args=(1,), disabled=step == len(wt.STEPS) - 1,
                  use_container_width=True)
    nav[2].markdown(f"P(class 1) for the star: **{t['proba'][0]:.1%}** · forward passes: "
                    f"**1** · weight updates: **0**")
    show(wt.chain_figure(step))              # the whole model; the current step in black
    st.markdown(f"**{step + 1} / {len(wt.STEPS)} · {wt.STEPS[step]}** — {wt.step_text(step)}")
    show(wt.step_figure(step, t, x_train, y_train, (x1, x2),
                        walk_surfaces(split)[WALK_PRIOR["steps"]]))
