import argparse
import datetime
import hashlib
import json
import os
import platform
import shutil

import torch

import run_demo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cpu_name():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def main():
    parser = argparse.ArgumentParser(description="Создаёт эталон reference/ для test_reference.py")
    parser.add_argument("--video", default=os.path.join(ROOT, "tests", "data", "apple.mp4"))
    parser.add_argument("--out", default=os.path.join(ROOT, "reference"))
    args = parser.parse_args()

    name = os.path.splitext(os.path.basename(args.video))[0]
    tmp = os.path.join(run_demo.DEFAULT_RESULTS, "reference_run")
    summary, paths = run_demo.run(args.video, out_dir=tmp, grid_size=10)

    os.makedirs(args.out, exist_ok=True)
    shutil.copyfile(paths["tracks"], os.path.join(args.out, f"{name}_tracks.npy"))
    shutil.copyfile(paths["visibility"], os.path.join(args.out, f"{name}_visibility.npy"))
    shutil.copyfile(paths["png"], os.path.join(args.out, f"{name}_summary.png"))
    meta = {
        "video": os.path.relpath(args.video, ROOT),
        "video_sha256": sha256_of(args.video),
        "checkpoint_sha256": sha256_of(run_demo.DEFAULT_CHECKPOINT),
        "grid_size": summary["grid_size"],
        "num_frames": summary["num_frames"],
        "num_points": summary["num_points"],
        "frame_size": summary["frame_size"],
        "visible_fraction": summary["visible_fraction"],
        "torch_version": torch.__version__,
        "torch_threads": summary["torch_threads"],
        "cpu": cpu_name(),
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    with open(os.path.join(args.out, f"{name}_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"reference saved to {args.out}: {name}_tracks.npy, {name}_visibility.npy, "
          f"{name}_summary.png, {name}_meta.json")


if __name__ == "__main__":
    main()
