import argparse
import functools
import json
import math
import os
import resource
import time
import warnings

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from cotracker.predictor import CoTrackerPredictor
from cotracker.utils.visualizer import Visualizer

DEFAULT_CHECKPOINT = os.environ.get("COTRACKER_CHECKPOINT", "./checkpoints/scaled_offline.pth")
DEFAULT_RESULTS = os.environ.get("RESULTS_DIR", "./results")


def log(msg):
    print(msg, flush=True)


def cpu_quota():
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            quota, period = f.read().split()
        if quota != "max":
            return max(1, math.ceil(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
            quota = int(f.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
            period = int(f.read())
        if quota > 0:
            return max(1, math.ceil(quota / period))
    except (OSError, ValueError):
        pass
    return None


def setup_threads(threads=0):
    if threads <= 0:
        threads = torch.get_num_threads()
        quota = cpu_quota()
        if quota:
            threads = min(threads, quota)
    torch.set_num_threads(threads)
    return threads


def pick_device(device="auto"):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def load_model(checkpoint=DEFAULT_CHECKPOINT, device="cpu", fmaps_chunk_size=8):
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*weights_only.*")
        model = CoTrackerPredictor(checkpoint=checkpoint)
    if fmaps_chunk_size:
        model.model.forward = functools.partial(model.model.forward,
                                                fmaps_chunk_size=fmaps_chunk_size)
    return model.to(device)


def read_video(path, max_frames=0, resize=0):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"video not found: {path}")
    reader = imageio.get_reader(path)
    fps = float(reader.get_meta_data().get("fps") or 10.0)
    frames, source_size = [], None
    for frame in reader:
        frame = np.asarray(frame)[..., :3]
        if source_size is None:
            source_size = (frame.shape[1], frame.shape[0])
        if resize and resize != frame.shape[0]:
            h = int(round(resize / 2)) * 2
            w = int(round(frame.shape[1] * resize / frame.shape[0] / 2)) * 2
            frame = np.asarray(Image.fromarray(frame).resize((w, h), Image.BILINEAR))
        frames.append(frame)
        if max_frames and len(frames) >= max_frames:
            break
    reader.close()
    if not frames:
        raise ValueError(f"no frames in {path}")
    return np.stack(frames), fps, source_size


def track(model, frames, grid_size=10, device="cpu"):
    video = torch.from_numpy(frames).permute(0, 3, 1, 2)[None].float().to(device)
    tracks, visibility = model(video, grid_size=grid_size)
    return tracks[0].cpu().numpy().astype(np.float32), visibility[0].cpu().numpy().astype(bool)


def render_tracks_video(frames, tracks, visibility, trace=15, linewidth=2):
    vis = Visualizer(pad_value=0, linewidth=linewidth, show_first_frame=0,
                     tracks_leave_trace=trace)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = vis.visualize(
            torch.from_numpy(frames).permute(0, 3, 1, 2)[None].float(),
            torch.from_numpy(tracks)[None],
            torch.from_numpy(visibility)[None],
            save_video=False,
        )
    return out[0].permute(0, 2, 3, 1).numpy()


def write_mp4(path, frames, fps):
    writer = imageio.get_writer(
        path, fps=fps, codec="libx264", quality=None, macro_block_size=2,
        ffmpeg_params=["-crf", "20", "-preset", "veryfast", "-movflags", "+faststart"],
        ffmpeg_log_level="error",
    )
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def write_gif(path, frames, fps, width=400, max_frames=60, colors=64):
    step = max(1, math.ceil(len(frames) / max_frames))
    picked = frames[::step]
    h, w = picked.shape[1:3]
    size = (width, int(round(h * width / w))) if w > width else (w, h)
    images = [Image.fromarray(f).resize(size, Image.BILINEAR)
              .quantize(colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
              for f in picked]
    images[0].save(path, save_all=True, append_images=images[1:], loop=0,
                   duration=int(round(1000 * step / fps)), optimize=True)


def write_summary_png(path, frames, tracks, visibility):
    T, N = visibility.shape
    h, w = frames.shape[1:3]
    y0 = tracks[0, :, 1]
    colors = plt.get_cmap("gist_rainbow")((y0 - y0.min()) / max(np.ptp(y0), 1e-6))

    fig_w = 14.0
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_w / 2 * h / w + 0.8), dpi=100)
    axes[0].imshow(frames[0])
    axes[0].scatter(tracks[0, :, 0], tracks[0, :, 1], s=30, c=colors,
                    edgecolors="black", linewidths=0.6)
    axes[0].set_title(f"Кадр 1 из {T}: {N} точек запроса")

    axes[1].imshow(frames[-1])
    visible_xy = np.where(visibility[..., None], tracks, np.nan)
    for n in range(N):
        axes[1].plot(visible_xy[:, n, 0], visible_xy[:, n, 1], color=colors[n],
                     linewidth=1.3, alpha=0.9)
    last_vis = visibility[-1]
    axes[1].scatter(tracks[-1, last_vis, 0], tracks[-1, last_vis, 1], s=30,
                    c=colors[last_vis], edgecolors="black", linewidths=0.6)
    axes[1].scatter(tracks[-1, ~last_vis, 0], tracks[-1, ~last_vis, 1], s=30,
                    facecolors="none", edgecolors=colors[~last_vis], linewidths=1.5)
    axes[1].set_title(f"Кадр {T} из {T}: траектории (видимы {last_vis.sum()} из {N})")

    for ax in axes:
        ax.set_xlim(0, w - 1)
        ax.set_ylim(h - 1, 0)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run(video_path, out_dir=None, grid_size=10, max_frames=0, resize=0,
        checkpoint=DEFAULT_CHECKPOINT, device="auto", threads=0, trace=15,
        gif_width=400, fmaps_chunk_size=8, model=None):
    t_start = time.time()
    device = pick_device(device)
    threads = setup_threads(threads)
    name = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = out_dir or os.path.join(DEFAULT_RESULTS, "demo", name)
    os.makedirs(out_dir, exist_ok=True)

    log(f"[1/5] Reading video {video_path}")
    frames, fps, source_size = read_video(video_path, max_frames, resize)
    T, H, W = frames.shape[:3]
    log(f"      {T} frames, {W}x{H} (source {source_size[0]}x{source_size[1]}), {fps:g} fps")

    log(f"[2/5] Loading model {checkpoint} on {device} ({threads} CPU threads)")
    t0 = time.time()
    if model is None:
        model = load_model(checkpoint, device, fmaps_chunk_size)
    load_seconds = time.time() - t0

    log(f"[3/5] Tracking a {grid_size}x{grid_size} grid of points")
    t0 = time.time()
    tracks, visibility = track(model, frames, grid_size, device)
    infer_seconds = time.time() - t0
    log(f"      tracks {tracks.shape}, visibility {visibility.shape}, {infer_seconds:.1f} s")

    log("[4/5] Rendering mp4 / gif / png")
    t0 = time.time()
    rendered = render_tracks_video(frames, tracks, visibility, trace=trace)
    paths = {
        "video": os.path.join(out_dir, "tracks.mp4"),
        "gif": os.path.join(out_dir, "preview.gif"),
        "png": os.path.join(out_dir, "summary.png"),
        "tracks": os.path.join(out_dir, "tracks.npy"),
        "visibility": os.path.join(out_dir, "visibility.npy"),
        "summary": os.path.join(out_dir, "summary.json"),
    }
    write_mp4(paths["video"], rendered, fps)
    write_gif(paths["gif"], rendered, fps, width=gif_width)
    write_summary_png(paths["png"], frames, tracks, visibility)
    render_seconds = time.time() - t0

    log("[5/5] Saving arrays and summary.json")
    np.save(paths["tracks"], tracks)
    np.save(paths["visibility"], visibility)
    vis_pts = visibility[-1] & visibility[0]
    displacement = np.linalg.norm(tracks[-1] - tracks[0], axis=-1)
    summary = {
        "video": video_path,
        "source_size": list(source_size),
        "frame_size": [W, H],
        "fps": fps,
        "num_frames": T,
        "grid_size": grid_size,
        "num_points": int(tracks.shape[1]),
        "visible_fraction": round(float(visibility.mean()), 4),
        "visible_fraction_last_frame": round(float(visibility[-1].mean()), 4),
        "mean_displacement_px": round(float(displacement[vis_pts].mean()), 2) if vis_pts.any() else None,
        "model_load_seconds": round(load_seconds, 2),
        "inference_seconds": round(infer_seconds, 2),
        "render_seconds": round(render_seconds, 2),
        "total_seconds": round(time.time() - t_start, 2),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        "device": device,
        "torch_threads": threads,
        "torch_version": torch.__version__,
        "checkpoint": checkpoint,
        "outputs": {k: os.path.basename(v) for k, v in paths.items()},
    }
    with open(paths["summary"], "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log(f"      points: {summary['num_points']}, frames: {T}, "
        f"visible: {100 * summary['visible_fraction']:.1f}%, "
        f"inference: {infer_seconds:.1f} s, total: {summary['total_seconds']:.1f} s")
    log(f"      results: {out_dir}")
    return summary, paths


def main():
    parser = argparse.ArgumentParser(description="CoTracker3 demo: video -> tracks -> mp4/gif/png/npy/json")
    parser.add_argument("--video", default="assets/apple.mp4", help="путь к видео")
    parser.add_argument("--grid_size", type=int, default=10, help="сетка G x G точек на первом кадре")
    parser.add_argument("--max_frames", type=int, default=0, help="взять первые N кадров (0 — все)")
    parser.add_argument("--resize", type=int, default=0, help="высота кадра в пикселях (0 — исходная)")
    parser.add_argument("--out", default=None, help="папка результатов (по умолчанию $RESULTS_DIR/demo/<имя>)")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    parser.add_argument("--threads", type=int, default=0, help="потоки torch (0 — по квоте CPU)")
    parser.add_argument("--trace", type=int, default=15, help="длина хвоста трека в mp4/gif, кадров")
    parser.add_argument("--gif_width", type=int, default=400)
    parser.add_argument("--fmaps_chunk", type=int, default=8,
                        help="кадров за проход энкодера признаков (меньше — меньше память)")
    args = parser.parse_args()
    run(args.video, args.out, args.grid_size, args.max_frames, args.resize,
        args.checkpoint, args.device, args.threads, args.trace, args.gif_width,
        args.fmaps_chunk)


if __name__ == "__main__":
    main()
