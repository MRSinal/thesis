"""Pre-extract JPEG frames from surgical videos.

Usage:
    python extract_frames.py \
        --video-roots ../PitVis/videos \
        --out ../PitVis/frames \
        --img-size 224 \
        --every 1 \
        --jobs 16

Layout produced:
    <out>/<video_stem>/frame_000001.jpg
    <out>/<video_stem>/frame_000002.jpg
    ...
    <out>/<video_stem>/.done       (marker)

Notes:
    --every N keeps every Nth source frame. Set to match your training frameskip
    to cut storage ~Nx; then use frameskip=1 in the dataset. Leave at 1 for
    flexibility across frameskip experiments.
"""

import argparse
import concurrent.futures as cf
import os
import subprocess
from pathlib import Path


VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".m4v"}


def extract_one(video_path: Path, out_dir: Path, img_size: int, quality: int, every: int):
    vid_out = out_dir / video_path.stem
    done_marker = vid_out / ".done"
    if done_marker.exists():
        return f"skip   {video_path.name}"

    vid_out.mkdir(parents=True, exist_ok=True)

    vf = f"scale={img_size}:{img_size}:flags=bilinear"
    if every > 1:
        vf = f"select='not(mod(n\\,{every}))',{vf}"

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(video_path),
        "-vf", vf,
        "-vsync", "vfr",
        "-q:v", str(quality),
        str(vid_out / "frame_%06d.jpg"),
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        return f"FAIL   {video_path.name}: {e}"

    done_marker.touch()
    n = sum(1 for p in vid_out.iterdir() if p.suffix == ".jpg")
    return f"done   {video_path.name} ({n} frames)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-roots", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--quality", type=int, default=3,
                    help="ffmpeg -q:v (2=best, 31=worst); 3 is a good default")
    ap.add_argument("--every", type=int, default=1,
                    help="keep every Nth source frame (1 = all)")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = []
    for r in args.video_roots:
        root = Path(r)
        for v in sorted(root.iterdir()):
            if v.suffix.lower() in VIDEO_EXTS:
                videos.append(v)
    print(f"found {len(videos)} videos; writing frames to {out_dir} with {args.jobs} workers")

    with cf.ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futures = [
            ex.submit(extract_one, v, out_dir, args.img_size, args.quality, args.every)
            for v in videos
        ]
        for f in cf.as_completed(futures):
            print(f.result(), flush=True)


if __name__ == "__main__":
    main()
