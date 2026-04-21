import random
import os
import shutil
import argparse
import cv2


def frame_count(path):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--max-frames", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="subset.txt")
    parser.add_argument("--out-dir", help="if set, copy selected videos here")
    parser.add_argument("--symlink", action="store_true", help="symlink into --out-dir instead of copy")
    args = parser.parse_args()

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    random.seed(args.seed)
    videos = sorted(os.listdir(args.path))
    random.shuffle(videos)

    selected = []
    total = 0
    for name in videos:
        n = frame_count(os.path.join(args.path, name))
        if n <= 0:
            continue
        if total + n > args.max_frames:
            continue
        selected.append((name, n))
        total += n
        if total >= args.max_frames:
            break

    with open(args.out, "w") as f:
        for name, n in selected:
            f.write(f"{name}\t{n}\n")

    if args.out_dir:
        for name, _ in selected:
            src = os.path.abspath(os.path.join(args.path, name))
            dst = os.path.join(args.out_dir, name)
            if os.path.lexists(dst):
                os.remove(dst)
            if args.symlink:
                os.symlink(src, dst)
            else:
                shutil.copy2(src, dst)

    print(f"Selected {len(selected)}/{len(videos)} videos, {total} frames -> {args.out}")
    if args.out_dir:
        print(f"{'Linked' if args.symlink else 'Copied'} to {args.out_dir}")


if __name__ == "__main__":
    main()
