import json
import os

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
import pytest
from PIL import Image

from helpers import DATA_DIR, RESULTS_DIR, plot_trajectories, predict, rel, run_demo


def test_demo_on_real_video(apple_demo, report):
    summary, paths = apple_demo
    source = os.path.join(DATA_DIR, "apple.mp4")
    n_in, _ = imageio_ffmpeg.count_frames_and_secs(source)

    for key, path in paths.items():
        assert os.path.isfile(path), f"нет файла {key}: {path}"
        assert os.path.getsize(path) > 0, f"пустой файл {path}"

    n_out, _ = imageio_ffmpeg.count_frames_and_secs(paths["video"])
    assert n_out == n_in == summary["num_frames"], f"кадров в mp4 {n_out}, во входе {n_in}"
    reader = imageio.get_reader(paths["video"])
    out_size = tuple(reader.get_meta_data()["size"])
    reader.close()
    assert out_size == tuple(summary["frame_size"])

    with Image.open(paths["gif"]) as gif:
        assert gif.n_frames == n_in
    with Image.open(paths["png"]) as png:
        assert png.width > 500 and png.height > 200

    tracks, visibility = np.load(paths["tracks"]), np.load(paths["visibility"])
    assert tracks.shape == (n_in, 100, 2) and visibility.shape == (n_in, 100)
    assert np.isfinite(tracks).all()

    with open(paths["summary"], encoding="utf-8") as f:
        saved = json.load(f)
    assert saved["num_points"] == 100 and saved["num_frames"] == n_in
    assert 0.5 < saved["visible_fraction"] <= 1.0

    out_dir = os.path.dirname(paths["video"])
    report(f"{n_in} frames -> mp4 {n_out} frames {out_size[0]}x{out_size[1]}, gif, png, npy, json; "
           f"visible {100 * saved['visible_fraction']:.1f}%, inference {saved['inference_seconds']:.1f} s",
           artifact=out_dir, frames=n_out, inference_seconds=saved["inference_seconds"])


@pytest.mark.parametrize("name", ["shapes", "pan_zoom"])
def test_procedural_video_accuracy(model, device, report, name):
    limit_err = 1.5
    frames, fps, _ = run_demo.read_video(os.path.join(DATA_DIR, f"{name}.mp4"))
    gt = np.load(os.path.join(DATA_DIR, f"{name}_gt.npz"))
    gt_tracks, gt_vis = gt["tracks"], gt["visibility"]
    assert len(frames) == len(gt_tracks), "число кадров mp4 не совпадает с разметкой"

    tracks, visibility = predict(model, frames, queries=gt["queries"], device=device)
    tracks, visibility = tracks[0].numpy(), visibility[0].numpy()

    err = np.linalg.norm(tracks - gt_tracks, axis=-1)
    mask = gt_vis.copy()
    mask[0] = False
    per_frame = np.array([err[t][mask[t]].mean() if mask[t].any() else np.nan
                          for t in range(len(err))])
    vis_acc = (visibility == gt_vis).mean()
    hidden_ok = (~visibility[~gt_vis]).mean()

    out_dir = os.path.join(RESULTS_DIR, f"e2e_{name}")
    os.makedirs(out_dir, exist_ok=True)
    rendered = run_demo.render_tracks_video(frames, tracks, visibility)
    run_demo.write_mp4(os.path.join(out_dir, "tracks.mp4"), rendered, fps)
    run_demo.write_gif(os.path.join(out_dir, "preview.gif"), rendered, fps)
    png = os.path.join(out_dir, "gt_vs_pred.png")
    plot_trajectories(png, frames[-1], tracks, visibility, gt=gt_tracks, errors=per_frame,
                      limit=limit_err,
                      title=f"{name}.mp4: ошибка {err[mask].mean():.3f} px, "
                            f"видимость угадана в {100 * vis_acc:.1f}% случаев")

    assert err[mask].mean() < limit_err, f"средняя ошибка {err[mask].mean():.2f} px"
    assert vis_acc >= 0.95
    assert hidden_ok >= 0.8, f"закрытыми признаны {100 * hidden_ok:.0f}% закрытых точек"
    report(f"error mean {err[mask].mean():.3f} px, p95 {np.percentile(err[mask], 95):.3f} px "
           f"(limit {limit_err:g}); visibility {100 * vis_acc:.1f}%, "
           f"hidden points detected {100 * hidden_ok:.0f}%",
           artifact=png, err_mean_px=float(err[mask].mean()),
           visibility_accuracy=float(vis_acc), hidden_detected=float(hidden_ok),
           video=rel(os.path.join(out_dir, "tracks.mp4")))
