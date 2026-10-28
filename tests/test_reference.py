import json
import os

import numpy as np
import pytest

from helpers import DATA_DIR, REFERENCE_DIR, RESULTS_DIR, plot_trajectories, run_demo

LIMIT_PX = 0.5


def test_matches_reference(apple_demo, report):
    ref_tracks_path = os.path.join(REFERENCE_DIR, "apple_tracks.npy")
    if not os.path.isfile(ref_tracks_path):
        pytest.skip("нет эталона reference/apple_tracks.npy: создайте его `bash run.sh --make-reference`")
    ref_tracks = np.load(ref_tracks_path)
    ref_vis = np.load(os.path.join(REFERENCE_DIR, "apple_visibility.npy"))
    with open(os.path.join(REFERENCE_DIR, "apple_meta.json"), encoding="utf-8") as f:
        meta = json.load(f)

    summary, paths = apple_demo
    assert meta["grid_size"] == summary["grid_size"]
    assert meta["num_frames"] == summary["num_frames"]
    tracks, visibility = np.load(paths["tracks"]), np.load(paths["visibility"])
    assert tracks.shape == ref_tracks.shape

    both = visibility & ref_vis
    err = np.linalg.norm(tracks - ref_tracks, axis=-1)
    per_frame = np.array([err[t][both[t]].mean() if both[t].any() else np.nan
                          for t in range(len(err))])
    agreement = (visibility == ref_vis).mean()

    frames, _, _ = run_demo.read_video(os.path.join(DATA_DIR, "apple.mp4"))
    path = os.path.join(RESULTS_DIR, "reference_compare.png")
    plot_trajectories(path, frames[-1], tracks, visibility, gt=ref_tracks, errors=per_frame,
                      limit=LIMIT_PX, gt_label="эталон",
                      title=f"Сравнение с reference/ ({meta.get('cpu', '?')}): "
                            f"расхождение {err[both].mean():.4f} px")

    assert err[both].mean() < LIMIT_PX, f"среднее расхождение {err[both].mean():.3f} px"
    assert agreement >= 0.98, f"видимость совпадает в {100 * agreement:.1f}% случаев"
    report(f"mean diff {err[both].mean():.4f} px, max {err[both].max():.4f} px "
           f"(limit {LIMIT_PX:g}); visibility agreement {100 * agreement:.2f}%",
           artifact=path, diff_mean_px=float(err[both].mean()),
           diff_max_px=float(err[both].max()), visibility_agreement=float(agreement))
