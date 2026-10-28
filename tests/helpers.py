import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TESTS_DIR)
DATA_DIR = os.path.join(TESTS_DIR, "data")
REFERENCE_DIR = os.path.join(ROOT, "reference")
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import run_demo

RESULTS_DIR = os.path.join(run_demo.DEFAULT_RESULTS, "tests")


def rel(path):
    base = os.path.dirname(os.path.abspath(run_demo.DEFAULT_RESULTS))
    return os.path.relpath(os.path.abspath(path), base)


def predict(model, frames, queries=None, grid_size=0, device="cpu"):
    assert queries is not None or grid_size > 0, "без queries и grid_size модель считает плотные треки"
    video = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2)[None].float().to(device)
    q = None if queries is None else torch.as_tensor(queries, dtype=torch.float32)[None].to(device)
    tracks, visibility = model(video, queries=q, grid_size=grid_size)
    return tracks.cpu(), visibility.cpu()


def plot_trajectories(path, frame, pred, pred_vis, gt=None, title="", errors=None, limit=None,
                      gt_label="истина"):
    H, W = frame.shape[:2]
    ncols = 1 if errors is None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6.5 * ncols, 6.5 * H / W + 0.9), dpi=100,
                             squeeze=False)
    ax = axes[0, 0]
    ax.imshow(frame)
    pred_xy = np.where(pred_vis[..., None], pred, np.nan)
    for n in range(pred.shape[1]):
        if gt is not None:
            ax.plot(gt[:, n, 0], gt[:, n, 1], color="lime", linewidth=2.2, alpha=0.8)
            ax.plot(gt[-1, n, 0], gt[-1, n, 1], "x", color="lime", markersize=8, mew=2)
        ax.plot(pred_xy[:, n, 0], pred_xy[:, n, 1], "--", color="red", linewidth=1.2)
        ax.plot(pred[-1, n, 0], pred[-1, n, 1], "o", markersize=5,
                color="red" if pred_vis[-1, n] else "none", markeredgecolor="red")
    handles = [Line2D([], [], color="red", linestyle="--", marker="o", label="предсказание")]
    if gt is not None:
        handles.insert(0, Line2D([], [], color="lime", marker="x", label=gt_label))
    ax.legend(handles=handles, loc="upper right", fontsize=9)
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_title(f"последний кадр (t={pred.shape[0] - 1})")
    ax.axis("off")

    if errors is not None:
        ax = axes[0, 1]
        ax.plot(np.arange(len(errors)), errors, marker=".", color="tab:blue",
                label="средняя ошибка")
        if limit is not None:
            ax.axhline(limit, color="red", linestyle="--", label=f"порог {limit:g} px")
        ax.set_xlabel("кадр")
        ax.set_ylabel("ошибка, px")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
        ax.legend()
        ax.set_title("ошибка по кадрам (видимые точки)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_points_strip(path, frames, tracks, vis, picks, title="", boxes=None):
    H, W = frames.shape[1:3]
    fig, axes = plt.subplots(1, len(picks), figsize=(4 * len(picks), 4 * H / W + 0.8), dpi=100,
                             squeeze=False)
    for ax, t in zip(axes[0], picks):
        ax.imshow(frames[t])
        v = vis[t]
        ax.scatter(tracks[t, v, 0], tracks[t, v, 1], s=22, c="lime", edgecolors="black",
                   linewidths=0.5)
        ax.scatter(tracks[t, ~v, 0], tracks[t, ~v, 1], s=26, facecolors="none",
                   edgecolors="red", linewidths=1.6)
        if boxes and t in boxes:
            x0, y0, w, h = boxes[t]
            ax.add_patch(Rectangle((x0, y0), w, h, fill=False, edgecolor="yellow",
                                   linestyle="--", linewidth=1.5))
        ax.set_xlim(-0.5, W - 0.5)
        ax.set_ylim(H - 0.5, -0.5)
        ax.set_title(f"t={t}: видимы {v.sum()} из {len(v)}")
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
