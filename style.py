"""One look for every figure. Importing this module sets it.

The rule behind the colours: the interface itself is black, white and grey, so
every colour on the screen is data. Each model has one colour; our model,
mini-PFN B, is black and its panel gets a black frame. Blue = class 0,
red = class 1.

The typeface is Atkinson Hyperlegible (static/, SIL Open Font License). It was
designed to stay readable at a distance, which is what a projected demo needs.
The app uses the same files for its text (.streamlit/config.toml).
"""
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # draw into images only; no window (the app and tests run headless)
import matplotlib.pyplot as plt
from matplotlib import font_manager

INK, GREY, RULE = "#1c1c1c", "#5f625c", "#d5d7d2"
BLUE, RED = "#184f95", "#9f3332"
OURS = "Mini-PFN B"
COLOURS = {"Mini-PFN A": "#9a9da1", "Mini-PFN B": INK, "TabPFN v2 (official)": "#1baf7a",
           "Logistic regression": "#eda100", "MLP (gradient descent)": "#e87ba4",
           "Gradient-boosted trees": "#008300"}

for file in (Path(__file__).parent / "static").glob("*.ttf"):
    font_manager.fontManager.addfont(file)

plt.rcParams.update({
    "font.family": ["Atkinson Hyperlegible", "sans-serif"], "font.size": 11,
    "text.color": INK, "axes.labelcolor": GREY, "xtick.color": GREY, "ytick.color": GREY,
    "axes.edgecolor": RULE, "axes.linewidth": 0.8, "xtick.major.size": 0, "ytick.major.size": 0,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "savefig.dpi": 160,
})


def headline(ax, name, number):
    """Two texts above a panel: the model (wrapped after about 14 characters)
    and its key number in large type."""
    ax.text(0, 1.17, textwrap.fill(name, 14), transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="bottom", linespacing=1.1)
    ax.text(0, 1.02, number, transform=ax.transAxes, fontsize=22, fontweight="bold")


def frame(ax, name):
    """A black frame marks the panel of our own model, a light one the others."""
    for side in ax.spines.values():
        side.set_visible(True)
        side.set(color=INK if name == OURS else RULE, linewidth=2 if name == OURS else 0.8)
