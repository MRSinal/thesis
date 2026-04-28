"""Cholec80 single-frame phase classification dataset.

Layout:
    <frames_root>/videoXX/videoXX_NNNNNN.png   (1-indexed, extracted at 1 fps)
    <phase_root>/videoXX-phase.txt              ("Frame\tPhase" rows at 25 fps)

Mapping: extracted file index N (1-indexed) -> annotation row (N - 1) * 25.
Phase boundaries are minute-scale, so any sub-second misalignment is irrelevant.

Splits videos by name (sorted -> torch.randperm(seed) -> partition at
train_ratio). Per-frame splitting would leak adjacent frames across train/val.
"""

from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, Dataset
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import v2 as T

FRAME_EXTS = {".png", ".jpg", ".jpeg"}
SOURCE_FPS_RATIO = 25

PHASE_TO_IDX = {
    "Preparation": 0,
    "CalotTriangleDissection": 1,
    "ClippingCutting": 2,
    "GallbladderDissection": 3,
    "GallbladderPackaging": 4,
    "CleaningCoagulation": 5,
    "GallbladderRetraction": 6,
}


def _load_phases(path):
    labels = []
    with open(path) as f:
        next(f)  # skip header
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                labels.append(PHASE_TO_IDX[parts[1]])
    return labels


def _build_transform(img_size):
    return T.Compose([
        T.Resize((img_size, img_size), antialias=True),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class _VideoFrames(Dataset):
    """All labeled frames from one video."""

    def __init__(self, vid_dir: Path, phase_root: Path, transform):
        rows = _load_phases(phase_root / f"{vid_dir.name}-phase.txt")
        frames = sorted(
            p for p in vid_dir.iterdir() if p.suffix.lower() in FRAME_EXTS
        )
        self.items = []
        for f in frames:
            n = int(f.stem.split("_")[-1])
            row = (n - 1) * SOURCE_FPS_RATIO
            if row < len(rows):
                self.items.append((str(f), rows[row]))
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        img = read_image(path, ImageReadMode.RGB)
        return {"pixels": self.transform(img), "label": label}


class Cholec80Dataset(ConcatDataset):
    """Cholec80 frames + phase labels with by-video train/val split."""

    def __init__(
        self,
        frames_root="/home/u941663/thesis/data/cholec80/frames",
        phase_root="/home/u941663/thesis/data/cholec80/phase_annotations",
        split="all",
        train_ratio=0.6,
        img_size=224,
        seed=3072,
    ):
        assert split in ("train", "val", "all")
        frames_root = Path(frames_root)
        phase_root = Path(phase_root)

        videos = sorted(p for p in frames_root.iterdir() if p.is_dir())
        g = torch.Generator().manual_seed(seed)
        order = torch.randperm(len(videos), generator=g).tolist()
        n_train = round(train_ratio * len(videos))
        if split == "train":
            chosen = [videos[i] for i in order[:n_train]]
        elif split == "val":
            chosen = [videos[i] for i in order[n_train:]]
        else:
            chosen = videos

        transform = _build_transform(img_size)
        per_video, skipped = [], 0
        for v in chosen:
            if not (phase_root / f"{v.name}-phase.txt").exists():
                skipped += 1
                continue
            per_video.append(_VideoFrames(v, phase_root, transform))

        super().__init__(per_video)
        msg = (
            f"Cholec80Dataset[{split}]: {len(self)} frames from "
            f"{len(per_video)}/{len(chosen)} videos"
        )
        if skipped:
            msg += f" ({skipped} skipped: missing phase file)"
        print(msg)
