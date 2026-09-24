"""Qualitative figures shared by all phases."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HEIGHT_CMAP = "magma"
ERROR_CMAP = "RdBu_r"


def tile_panel(out: Path, tid: str, rgb, gt, valid, raw, preds: dict[str, np.ndarray],
               err_key: str, err_lim: float = 10.0, title_extra: str = "") -> None:
    """RGB | GT | raw relative | preds... | error(err_key - GT). Heights share one scale."""
    vmax = max(1.0, float(np.percentile(gt[valid], 99.5)))
    n = 3 + len(preds) + 1
    fig, ax = plt.subplots(1, n, figsize=(3.6 * n, 4.2))
    ax[0].imshow(rgb); ax[0].set_title(f"{tid} RGB")
    ax[1].imshow(np.where(valid, gt, np.nan), cmap=HEIGHT_CMAP, vmin=0, vmax=vmax); ax[1].set_title("GT nDSM (m)")
    ax[2].imshow(raw, cmap="Greys_r"); ax[2].set_title("DA-V2-S raw (relative)")
    for a, (k, p) in zip(ax[3:], preds.items()):
        im = a.imshow(p, cmap=HEIGHT_CMAP, vmin=0, vmax=vmax); a.set_title(k, fontsize=9)
    fig.colorbar(im, ax=ax[3 + len(preds) - 1], fraction=0.046, label="m")
    e = np.where(valid, preds[err_key] - gt, np.nan)
    im = ax[-1].imshow(e, cmap=ERROR_CMAP, vmin=-err_lim, vmax=err_lim)
    ax[-1].set_title(f"error {err_key.split(' ')[0]} - GT", fontsize=9)
    fig.colorbar(im, ax=ax[-1], fraction=0.046, label="m")
    for a in ax: a.axis("off")
    if title_extra:
        fig.suptitle(title_extra, fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=80); plt.close(fig)


def density_scatter(out: Path, pairs: dict[str, tuple[np.ndarray, np.ndarray]], lim: float = 40.0) -> None:
    """pairs: name -> (gt, pred). Log-density hexbin with the 1:1 line."""
    fig, ax = plt.subplots(1, len(pairs), figsize=(5.2 * len(pairs), 5))
    ax = np.atleast_1d(ax)
    for a, (name, (g, p)) in zip(ax, pairs.items()):
        a.hexbin(g, p, gridsize=80, extent=(0, lim, 0, lim), bins="log", cmap="viridis", mincnt=1)
        a.plot([0, lim], [0, lim], "r--", lw=1)
        a.set_xlabel("GT nDSM (m)"); a.set_ylabel("predicted (m)"); a.set_title(name, fontsize=10)
        a.set_xlim(0, lim); a.set_ylim(0, lim)
    fig.tight_layout(); fig.savefig(out, dpi=80); plt.close(fig)


def grouped_bars(out: Path, table: dict[str, dict[str, float]], ylabel: str, title: str) -> None:
    """table: method -> {group: value}. One bar group per group, one bar per method."""
    methods = list(table)
    groups = [g for g in next(iter(table.values()))]
    x = np.arange(len(groups)); w = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(max(7, 1.3 * len(groups) * len(methods) / 2), 4.5))
    for i, m in enumerate(methods):
        ax.bar(x + i * w - 0.4 + w / 2, [table[m].get(g) or 0 for g in groups], w, label=m)
    ax.set_xticks(x); ax.set_xticklabels(groups, rotation=20)
    ax.set_ylabel(ylabel); ax.set_title(title); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=80); plt.close(fig)


def calibration_curve(out: Path, x: np.ndarray, y: np.ndarray, curves: dict[str, tuple], title: str) -> None:
    """Train-pixel density of normalised relative depth vs GT nDSM with fitted maps."""
    fig, ax = plt.subplots(figsize=(6.5, 5))
    lo, hi = np.percentile(x, [0.5, 99.5])
    ax.hexbin(x, y, gridsize=90, extent=(lo, hi, 0, 40), bins="log", cmap="Greys", mincnt=1)
    xs = np.linspace(lo, hi, 400)
    for name, (f, style) in curves.items():
        ax.plot(xs, f(xs), style, lw=2, label=name)
    ax.set_xlim(lo, hi); ax.set_ylim(0, 40)
    ax.set_xlabel("normalised relative inverse depth d~"); ax.set_ylabel("GT nDSM (m)")
    ax.set_title(title, fontsize=10); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=80); plt.close(fig)
