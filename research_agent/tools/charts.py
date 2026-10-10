"""Matplotlib chart helpers with one consistent house style."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PALETTE = ["#1F4E79", "#E07B39", "#2A9D8F", "#B5365A", "#6C757D", "#8E6BBF", "#C9A227", "#3A86FF"]


def _style():
    plt.rcParams.update({
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.alpha": 0.25, "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
        "font.size": 9, "axes.titlesize": 11, "axes.titleweight": "bold",
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def _save(fig, path: str) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _short(v, n=28) -> str:
    s = str(v)
    return s if len(s) <= n else s[: n - 1] + "..."


def line_chart(df: pd.DataFrame, x: str, y: str, group: str | None, title: str, path: str, ylabel: str = "") -> str:
    _style()
    fig, ax = plt.subplots(figsize=(8, 4.4))
    if group and group in df.columns:
        for g, sub in df.groupby(group):
            sub = sub.sort_values(x)
            ax.plot(sub[x], sub[y], marker="o" if len(sub) < 25 else None, ms=3, lw=1.8, label=_short(g, 22))
        ax.legend(frameon=False, fontsize=8, ncol=2 if df[group].nunique() > 3 else 1)
    else:
        d = df.sort_values(x)
        ax.plot(d[x], d[y], lw=2, marker="o" if len(d) < 25 else None, ms=3)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(ylabel or y)
    return _save(fig, path)


def bar_chart(labels, values, title: str, path: str, xlabel: str = "", horizontal: bool = True) -> str:
    _style()
    labels = [_short(l, 30) for l in labels]
    fig, ax = plt.subplots(figsize=(8, max(3.2, 0.38 * len(labels) + 1.2)) if horizontal else (8, 4.2))
    if horizontal:
        bars = ax.barh(labels[::-1], list(values)[::-1], color=PALETTE[0])
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", visible=False)
        for b in bars:
            w = b.get_width()
            ax.text(w, b.get_y() + b.get_height() / 2, f" {w:,.3g}", va="center", fontsize=8)
    else:
        bars = ax.bar(labels, values, color=PALETTE[0])
        ax.set_ylabel(xlabel)
        ax.grid(axis="x", visible=False)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_title(title)
    return _save(fig, path)


def hist_chart(series: pd.Series, title: str, path: str, xlabel: str = "") -> str:
    _style()
    s = series.dropna()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(s, bins=min(40, max(8, int(np.sqrt(len(s))))), color=PALETTE[0], alpha=0.9, edgecolor="white")
    ax.axvline(s.mean(), color=PALETTE[1], lw=1.6, ls="--", label=f"mean {s.mean():,.3g}")
    ax.axvline(s.median(), color=PALETTE[2], lw=1.6, ls=":", label=f"median {s.median():,.3g}")
    ax.legend(frameon=False)
    ax.set_title(title)
    ax.set_xlabel(xlabel or series.name)
    ax.set_ylabel("count")
    return _save(fig, path)


def scatter_chart(df: pd.DataFrame, x: str, y: str, title: str, path: str) -> str:
    _style()
    d = df[[x, y]].dropna()
    if len(d) > 3000:
        d = d.sample(3000, random_state=0)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.scatter(d[x], d[y], s=14, alpha=0.55, color=PALETTE[0], edgecolor="none")
    if len(d) > 2 and d[x].nunique() > 1:
        m, b = np.polyfit(d[x], d[y], 1)
        xs = np.linspace(d[x].min(), d[x].max(), 50)
        ax.plot(xs, m * xs + b, color=PALETTE[1], lw=2)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    return _save(fig, path)


def heatmap_chart(corr: pd.DataFrame, title: str, path: str) -> str:
    _style()
    n = len(corr)
    fig, ax = plt.subplots(figsize=(max(5, 0.8 * n + 2), max(4, 0.7 * n + 1.5)))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([_short(c, 16) for c in corr.columns], rotation=40, ha="right")
    ax.set_yticklabels([_short(c, 16) for c in corr.index])
    ax.grid(False)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(corr.values[i, j]) > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title)
    return _save(fig, path)
