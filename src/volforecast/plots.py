"""Static figures for the report (matplotlib, PNG)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Validated categorical order (blue, orange, aqua, ...) and recessive chrome.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)


def target_vs_vix(target_rv: pd.Series, vix: pd.Series, path: Path) -> None:
    """Forward 5-day realized vol of SPY against the VIX level, both in % p.a."""
    fig, ax = plt.subplots(figsize=(11, 4.2), facecolor=SURFACE)
    _style(ax)
    ax.plot(target_rv.index, 100 * target_rv, color=SERIES[0], linewidth=0.8,
            label="SPY realized vol, next 5 days (GK + overnight)")
    ax.plot(vix.index, vix, color=SERIES[1], linewidth=0.8, label="VIX (30-day implied)")
    ax.set_ylabel("Annualized volatility (%)", color=INK_2, fontsize=9)
    ax.set_ylim(bottom=0)
    ax.set_title("Target proxy vs implied volatility — SPY, 2005–2025",
                 loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
