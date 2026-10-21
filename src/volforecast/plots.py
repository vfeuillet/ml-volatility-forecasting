"""Static figures for the report (matplotlib, PNG)."""

from pathlib import Path

import matplotlib
import matplotlib.dates
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Validated categorical order (blue, orange, aqua, ...) and recessive chrome.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
GRID_DARK = "#8a8984"  # neutral for the realized series


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


# Fixed model -> colour mapping, shared by every figure (colour follows the model).
MODEL_COLORS = {"har": SERIES[0], "lstm": SERIES[1], "gbm": SERIES[2], "garch": SERIES[3],
                "gjr": SERIES[4], "harx_ticker": SERIES[5], "naive": SERIES[6],
                "harx_pooled": SERIES[7]}
MODEL_LABELS = {"har": "HAR", "lstm": "LSTM", "gbm": "Gradient boosting", "garch": "GARCH(1,1)",
                "gjr": "GJR-GARCH", "harx_ticker": "HAR-X (linear, ML features)", "naive": "Naive",
                "harx_pooled": "HAR-X pooled"}


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def forecasts_vs_realized(sample: pd.DataFrame, models: list[str], title: str, path: Path) -> None:
    """Realized 5-day vol against model forecasts (one ticker), in % p.a."""
    fig, ax = plt.subplots(figsize=(11, 4.2), facecolor=SURFACE)
    _style(ax)
    dates = sample.index
    ax.plot(dates, 100 * np.sqrt(sample["realized"]), color=GRID_DARK, linewidth=0.8,
            label="Realized (next 5 days)")
    for m in models:
        ax.plot(dates, 100 * np.sqrt(sample[m]), color=MODEL_COLORS[m], linewidth=1.2,
                label=MODEL_LABELS[m])
    ax.set_ylabel("Annualized volatility (%)", color=INK_2, fontsize=9)
    ax.set_ylim(bottom=0)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper left")
    _save(fig, path)


def cumulative_loss_difference(daily_diff: pd.DataFrame, path: Path) -> None:
    """Cumulative sum of the cross-sectional average QLIKE difference vs HAR.

    Going down = the model accumulates a lower loss than HAR.
    """
    fig, ax = plt.subplots(figsize=(11, 4.5), facecolor=SURFACE)
    _style(ax)
    ax.axhline(0, color=INK_2, linewidth=0.8)
    cum = daily_diff.cumsum()
    for m in cum.columns:
        ax.plot(cum.index, cum[m], color=MODEL_COLORS[m], linewidth=2, label=MODEL_LABELS[m])
        ax.annotate(MODEL_LABELS[m], (cum.index[-1], cum[m].iloc[-1]), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK_2)
    ax.set_ylabel("Cumulative QLIKE difference vs HAR", color=INK_2, fontsize=9)
    ax.set_title("Cumulative forecast loss relative to HAR (below 0 = better than HAR)",
                 loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper left")
    ax.set_xlim(cum.index[0], cum.index[-1] + pd.Timedelta(days=700))  # room for labels
    ax.set_xticks([pd.Timestamp(f"{y}-01-01") for y in range(cum.index[0].year, cum.index[-1].year + 2, 2)])
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    _save(fig, path)


def feature_importance(imp: pd.DataFrame, path: Path) -> None:
    """Permutation importance (mean and std across refits), horizontal bars."""
    imp = imp.sort_values("mean")
    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.barh(imp.index, imp["mean"], xerr=imp["std"], color=MODEL_COLORS["gbm"], height=0.6,
            error_kw={"ecolor": INK_2, "elinewidth": 0.8, "capsize": 2})
    ax.set_xlabel("Increase in validation MSE (log RV) when the feature is shuffled",
                  color=INK_2, fontsize=9)
    ax.set_title("Gradient boosting — permutation importance (mean ± sd over 168 refits)",
                 loc="left", color=INK, fontsize=11)
    _save(fig, path)


def backtest_nav(navs: pd.DataFrame, path: Path) -> None:
    """Growth of $1 (log scale) for vol-targeting strategies and buy-and-hold."""
    fig, ax = plt.subplots(figsize=(11, 4.8), facecolor=SURFACE)
    _style(ax)
    for col in navs.columns:
        color = MODEL_COLORS.get(col, GRID_DARK)
        style = "--" if col == "oracle" else "-"
        label = {"buy_hold": "SPY buy-and-hold",
                 "bh_matched": "SPY buy-and-hold scaled to the HAR strategy's vol (ex post)",
                 "oracle": "Oracle (realized vol, infeasible)"}.get(
            col, MODEL_LABELS.get(col, col))
        ax.plot(navs.index, navs[col], color=INK if col == "oracle" else color, linestyle=style,
                linewidth=1.2 if col in ("buy_hold", "bh_matched", "oracle") else 1.6, label=label)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([1, 1.5, 2, 3, 4, 6, 8]))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylabel("Growth of $1 (log scale)", color=INK_2, fontsize=9)
    ax.set_title("SPY volatility targeting (10% target, leverage ≤ 2, 5 bp costs, weekly)",
                 loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper left")
    _save(fig, path)
