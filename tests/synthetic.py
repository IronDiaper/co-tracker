import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CLIP_SIZE = (272, 480)
CLIP_FRAMES = 30
CLIP_FPS = 12


def make_texture(h, w, seed=0):
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.float32)
    for cell, weight in ((4, 1.0), (16, 0.6), (64, 0.35)):
        small = rng.random((max(2, h // cell), max(2, w // cell), 3)).astype(np.float32)
        img += weight * np.stack(
            [np.asarray(Image.fromarray(small[..., c]).resize((w, h), Image.BICUBIC))
             for c in range(3)],
            axis=-1,
        )
    img = (img - img.min()) / (img.max() - img.min()) * 255.0
    pil = Image.fromarray(img.astype(np.uint8))
    draw = ImageDraw.Draw(pil)
    for _ in range(h * w // 2500):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        r = int(rng.integers(3, 12))
        color = tuple(int(c) for c in rng.integers(0, 256, 3))
        if rng.random() < 0.5:
            draw.rectangle([x - r, y - r, x + r, y + r], fill=color)
        else:
            draw.ellipse([x - r, y - r, x + r, y + r], fill=color)
    return np.asarray(pil)


def make_shift_video(T=24, H=256, W=256, dx=3, dy=2, seed=0):
    pad = 8
    tex = make_texture(H + (T - 1) * abs(dy) + 2 * pad, W + (T - 1) * abs(dx) + 2 * pad, seed)
    ox0 = pad + max(0, (T - 1) * dx)
    oy0 = pad + max(0, (T - 1) * dy)
    frames = [tex[oy0 - t * dy: oy0 - t * dy + H, ox0 - t * dx: ox0 - t * dx + W]
              for t in range(T)]
    return np.stack(frames)


def shift_ground_truth(queries_xy, T, dx, dy):
    t = np.arange(T, dtype=np.float32)[:, None, None]
    return queries_xy[None].astype(np.float32) + t * np.array([dx, dy], np.float32)


def _visible_queries(ids0, points, own_ids):
    keep = ids0[points[:, 1], points[:, 0]] == own_ids
    return points[keep], own_ids[keep]


def make_shapes_clip(T=CLIP_FRAMES, size=CLIP_SIZE, seed=1):
    H, W = size
    s = 64
    bg = make_texture(H, W, seed)
    sprites = []
    for i in range(3):
        spr = make_texture(s, s, seed + 10 + i).copy()
        spr[:3], spr[-3:], spr[:, :3], spr[:, -3:] = 255, 255, 255, 255
        sprites.append(spr)
    pole = (0.35 * make_texture(H, 24, seed + 20)).astype(np.uint8)
    pole_x = 220

    t = np.arange(T)
    pos = [
        np.stack([20 + 10 * t, np.full(T, 30)], 1),
        np.stack([np.round(328 + 50 * np.cos(2 * np.pi * t / T)),
                  np.round(138 + 50 * np.sin(2 * np.pi * t / T))], 1).astype(int),
        np.stack([60 + 2 * t, np.round(150 + 40 * np.sin(2 * np.pi * t / T))], 1).astype(int),
    ]

    frames, ids = [], []
    for k in range(T):
        frame, idmap = bg.copy(), np.zeros((H, W), np.int32)
        for i, spr in enumerate(sprites):
            x, y = pos[i][k]
            frame[y:y + s, x:x + s] = spr
            idmap[y:y + s, x:x + s] = i + 1
        frame[:, pole_x:pole_x + pole.shape[1]] = pole
        idmap[:, pole_x:pole_x + pole.shape[1]] = 4
        frames.append(frame)
        ids.append(idmap)
    frames, ids = np.stack(frames), np.stack(ids)

    offsets = np.array([[ox, oy] for oy in (12, 32, 52) for ox in (12, 32, 52)])
    pts, own = [], []
    for i in range(3):
        pts.append(pos[i][0] + offsets)
        own.append(np.full(len(offsets), i + 1))
    gx, gy = np.meshgrid(np.linspace(16, W - 17, 8), np.linspace(16, H - 17, 5))
    pts.append(np.stack([gx.ravel(), gy.ravel()], 1).round().astype(int))
    own.append(np.zeros(gx.size, int))
    pts, own = _visible_queries(ids[0], np.concatenate(pts), np.concatenate(own))

    tracks = np.zeros((T, len(pts), 2), np.float32)
    vis = np.zeros((T, len(pts)), bool)
    for k in range(T):
        for n, (p, o) in enumerate(zip(pts, own)):
            q = p + (pos[o - 1][k] - pos[o - 1][0] if o > 0 else 0)
            tracks[k, n] = q
            inside = 0 <= q[0] < W and 0 <= q[1] < H
            vis[k, n] = inside and ids[k, int(q[1]), int(q[0])] == o
    queries = np.concatenate([np.zeros((len(pts), 1)), pts], 1).astype(np.float32)
    return frames, dict(queries=queries, tracks=tracks, visibility=vis)


def make_pan_zoom_clip(T=CLIP_FRAMES, size=CLIP_SIZE, seed=2):
    H, W = size
    tex = make_texture(int(2.5 * H), int(2.5 * W), seed)
    Ht, Wt = tex.shape[:2]
    tex_t = torch.from_numpy(tex.copy()).permute(2, 0, 1)[None].float()
    p_c = np.array([(W - 1) / 2, (H - 1) / 2])

    def camera(k):
        c = np.array([500.0, 320.0]) + k * np.array([4.0, 1.5])
        s, th = 1.0 + 0.01 * k, 0.004 * k
        M = s * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        return c, M

    ys, xs = np.mgrid[0:H, 0:W]
    pix = np.stack([xs, ys], -1).reshape(-1, 2) - p_c
    frames = []
    for k in range(T):
        c, M = camera(k)
        src = pix @ M.T + c
        grid = np.stack([2 * src[:, 0] / (Wt - 1) - 1, 2 * src[:, 1] / (Ht - 1) - 1], -1)
        grid = torch.from_numpy(grid.reshape(1, H, W, 2)).float()
        out = F.grid_sample(tex_t, grid, mode="bicubic", align_corners=True)
        frames.append(out[0].permute(1, 2, 0).clamp(0, 255).round().byte().numpy())
    frames = np.stack(frames)

    gx, gy = np.meshgrid(np.linspace(24, W - 25, 8), np.linspace(24, H - 25, 5))
    q = np.stack([gx.ravel(), gy.ravel()], 1)
    c0, M0 = camera(0)
    src = (q - p_c) @ M0.T + c0
    tracks = np.zeros((T, len(q), 2), np.float32)
    for k in range(T):
        c, M = camera(k)
        tracks[k] = (src - c) @ np.linalg.inv(M).T + p_c
    vis = ((tracks[..., 0] >= 0) & (tracks[..., 0] <= W - 1)
           & (tracks[..., 1] >= 0) & (tracks[..., 1] <= H - 1))
    queries = np.concatenate([np.zeros((len(q), 1)), q], 1).astype(np.float32)
    return frames, dict(queries=queries, tracks=tracks, visibility=vis)


def write_mp4(path, frames, fps):
    import imageio.v2 as imageio

    writer = imageio.get_writer(
        path, fps=fps, codec="libx264", quality=None, macro_block_size=16,
        ffmpeg_params=["-crf", "15", "-preset", "slow"], ffmpeg_log_level="error",
    )
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def draw_gt_preview(frames, gt, path, every=None):
    T = len(frames)
    picks = [0, T // 2, T - 1] if every is None else list(range(0, T, every))
    tiles = []
    for k in picks:
        img = Image.fromarray(frames[k])
        draw = ImageDraw.Draw(img)
        for n in range(gt["tracks"].shape[1]):
            path_xy = [tuple(gt["tracks"][j, n]) for j in range(k + 1)]
            if len(path_xy) > 1:
                draw.line(path_xy, fill=(255, 255, 0), width=1)
            x, y = gt["tracks"][k, n]
            r = 4
            if gt["visibility"][k, n]:
                draw.ellipse([x - r, y - r, x + r, y + r], fill=(255, 0, 0), outline=(0, 0, 0))
            else:
                draw.ellipse([x - r, y - r, x + r, y + r], outline=(255, 0, 0), width=2)
        draw.text((6, 4), f"t={k}", fill=(255, 255, 255))
        tiles.append(np.asarray(img))
    gap = np.full((frames.shape[1], 6, 3), 255, np.uint8)
    strip = np.concatenate(sum([[t, gap] for t in tiles], [])[:-1], axis=1)
    Image.fromarray(strip).save(path)


CLIPS = {"shapes": make_shapes_clip, "pan_zoom": make_pan_zoom_clip}


def main():
    parser = argparse.ArgumentParser(description="Пересоздаёт процедурные ролики tests/data/ с GT")
    parser.add_argument("--out", default=DATA_DIR)
    parser.add_argument("--preview", default=None, help="папка для PNG-превью с GT")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for name, make in CLIPS.items():
        frames, gt = make()
        mp4 = os.path.join(args.out, f"{name}.mp4")
        write_mp4(mp4, frames, CLIP_FPS)
        np.savez_compressed(os.path.join(args.out, f"{name}_gt.npz"), fps=CLIP_FPS, **gt)
        print(f"{name}: frames={frames.shape} points={gt['tracks'].shape[1]} "
              f"visible={gt['visibility'].mean():.3f} -> {mp4} ({os.path.getsize(mp4)} B)")
        if args.preview:
            os.makedirs(args.preview, exist_ok=True)
            draw_gt_preview(frames, gt, os.path.join(args.preview, f"{name}_gt.png"))


if __name__ == "__main__":
    main()
