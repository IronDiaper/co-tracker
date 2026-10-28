import os

import numpy as np
import pytest

import synthetic
from helpers import RESULTS_DIR, plot_points_strip, plot_trajectories, predict

T, S = 24, 256


def grid_queries(lo, hi, n=8):
    g = np.linspace(lo, hi, n)
    gx, gy = np.meshgrid(g, g)
    xy = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32)
    return np.concatenate([np.zeros((len(xy), 1), np.float32), xy], 1)


def test_static_video(model, device, report):
    limit_max, limit_mean = 1.0, 0.5
    frame = synthetic.make_texture(S, S, seed=3)
    video = np.repeat(frame[None], T, axis=0)
    tracks, visibility = predict(model, video, grid_size=8, device=device)
    tracks, visibility = tracks[0].numpy(), visibility[0].numpy()

    drift = np.linalg.norm(tracks - tracks[:1], axis=-1)[1:]
    path = os.path.join(RESULTS_DIR, "static_video.png")
    plot_trajectories(path, frame, tracks, visibility, gt=np.repeat(tracks[:1], T, 0),
                      errors=np.concatenate([[0.0], drift.mean(1)]), limit=limit_max,
                      title=f"Статичное видео: смещение в среднем {drift.mean():.3f} px, "
                            f"максимум {drift.max():.3f} px")

    assert drift.max() < limit_max, f"точка сместилась на {drift.max():.2f} px"
    assert drift.mean() < limit_mean
    assert visibility.mean() > 0.99
    report(f"drift mean {drift.mean():.3f} px, max {drift.max():.3f} px (limit {limit_max:g}); "
           f"visible {100 * visibility.mean():.1f}%",
           artifact=path, drift_mean_px=float(drift.mean()), drift_max_px=float(drift.max()))


@pytest.mark.parametrize("dx,dy", [(3, 2), (-2, 4)], ids=["dx3_dy2", "dx-2_dy4"])
def test_shift_accuracy(model, device, report, dx, dy):
    limit_err, limit_vel = 1.0, 0.1
    video = synthetic.make_shift_video(T, S, S, dx, dy, seed=4)
    queries = grid_queries(24, S - 25)
    gt = synthetic.shift_ground_truth(queries[:, 1:], T, dx, dy)
    gt_vis = (gt[..., 0] >= 0) & (gt[..., 0] <= S - 1) & (gt[..., 1] >= 0) & (gt[..., 1] <= S - 1)

    tracks, visibility = predict(model, video, queries=queries, device=device)
    tracks, visibility = tracks[0].numpy(), visibility[0].numpy()

    err = np.linalg.norm(tracks - gt, axis=-1)
    mask = gt_vis.copy()
    mask[0] = False
    per_frame = np.array([err[t][mask[t]].mean() if mask[t].any() else np.nan for t in range(T)])
    both = gt_vis[1:] & gt_vis[:-1]
    velocity = np.diff(tracks, axis=0)[both].mean(0)
    vel_err = np.abs(velocity - np.array([dx, dy])).max()
    vis_acc = (visibility == gt_vis).mean()

    path = os.path.join(RESULTS_DIR, f"shift_dx{dx}_dy{dy}.png")
    plot_trajectories(path, video[-1], tracks, visibility, gt=gt, errors=per_frame,
                      limit=limit_err,
                      title=f"Сдвиг ({dx}, {dy}) px/кадр: ошибка {err[mask].mean():.3f} px, "
                            f"скорость ({velocity[0]:.3f}, {velocity[1]:.3f})")

    assert err[mask].mean() < limit_err, f"средняя ошибка {err[mask].mean():.2f} px"
    assert vel_err < limit_vel, f"скорость {velocity} вместо ({dx}, {dy})"
    assert vis_acc > 0.95, f"видимость угадана в {100 * vis_acc:.1f}% случаев"
    report(f"error mean {err[mask].mean():.3f} px, p95 {np.percentile(err[mask], 95):.3f} px "
           f"(limit {limit_err:g}); velocity ({velocity[0]:.3f}, {velocity[1]:.3f}) "
           f"vs ({dx}, {dy}); visibility {100 * vis_acc:.1f}%",
           artifact=path, err_mean_px=float(err[mask].mean()), velocity_err=float(vel_err),
           visibility_accuracy=float(vis_acc))


def test_occlusion(model, device, report):
    x0, y0, size, t0, t1 = 72, 72, 112, 8, 16
    video = np.repeat(synthetic.make_texture(S, S, seed=5)[None], T, axis=0).copy()
    video[t0:t1, y0:y0 + size, x0:x0 + size] = synthetic.make_texture(size, size, seed=6)

    tracks, visibility = predict(model, video, grid_size=10, device=device)
    tracks, visibility = tracks[0].numpy(), visibility[0].numpy()
    q = tracks[0]
    covered = (q[:, 0] >= x0) & (q[:, 0] < x0 + size) & (q[:, 1] >= y0) & (q[:, 1] < y0 + size)
    gt_vis = np.ones_like(visibility)
    gt_vis[t0:t1, covered] = False

    hidden_ok = (~visibility[~gt_vis]).mean()
    visible_ok = visibility[gt_vis].mean()
    back = visibility[t1:, covered].mean()
    back_err = np.linalg.norm(tracks[t1:, covered] - q[covered], axis=-1).mean()

    path = os.path.join(RESULTS_DIR, "occlusion.png")
    plot_points_strip(path, video, tracks, visibility, picks=[t0 - 1, (t0 + t1) // 2, t1],
                      boxes={(t0 + t1) // 2: (x0, y0, size, size)},
                      title=f"Окклюзия: квадрат на кадрах {t0}–{t1 - 1} закрывает "
                            f"{covered.sum()} точек")

    assert covered.sum() >= 9
    assert hidden_ok >= 0.8, f"закрытыми признаны только {100 * hidden_ok:.0f}% закрытых точек"
    assert visible_ok >= 0.95
    assert back >= 0.8 and back_err < 1.0, "точки не нашлись после исчезновения квадрата"
    report(f"{covered.sum()} covered points: {100 * hidden_ok:.0f}% predicted hidden; "
           f"others visible {100 * visible_ok:.1f}%; after the square: visible "
           f"{100 * back:.0f}%, error {back_err:.3f} px",
           artifact=path, hidden_ok=float(hidden_ok), visible_ok=float(visible_ok),
           reappear_err_px=float(back_err))
