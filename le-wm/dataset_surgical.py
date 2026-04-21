import bisect
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.utils.data as data

# Avoid oversubscription: each DataLoader worker gets its own process,
cv2.setNumThreads(0)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

FRAME_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


class SurgicalVideoDataset(data.Dataset):
    """Loads consecutive frame sequences from pre-extracted JPEG frames.

    Expected layout under each root:
        root/
            video_a/
                frame_000001.jpg
                frame_000002.jpg
                ...
            video_b/...

    Each sample: {"pixels": (T, C, H, W) float tensor, ImageNet-normalized}.
    T = num_frames, with stride `frameskip` between sampled frames.
    """

    def __init__(self, video_roots, num_frames=4, frameskip=5, img_size=224):
        super().__init__()
        if isinstance(video_roots, str):
            video_roots = [video_roots]
        self.video_roots = video_roots
        self.num_frames = num_frames
        self.frameskip = frameskip
        self.img_size = img_size

        span = (num_frames - 1) * frameskip + 1

        self.videos = []  # list of (list[str paths], n_clips)
        self.cumulative_clips = []
        total = 0

        for root in video_roots:
            root = Path(root)
            for vid_dir in sorted(p for p in root.iterdir() if p.is_dir()):
                frames = sorted(
                    str(p) for p in vid_dir.iterdir()
                    if p.suffix.lower() in FRAME_EXTS
                )
                n_clips = max(0, len(frames) - span)
                if n_clips > 0:
                    self.videos.append((frames, n_clips))
                    total += n_clips
                    self.cumulative_clips.append(total)

        assert total > 0, f"No valid clips found in {video_roots}"
        print(f"SurgicalVideoDataset: {len(self.videos)} videos, {total} clips")

    def __len__(self):
        return self.cumulative_clips[-1]

    def __getitem__(self, idx):
        vid_idx = bisect.bisect_right(self.cumulative_clips, idx)
        frames_list, _ = self.videos[vid_idx]
        local_idx = idx - (self.cumulative_clips[vid_idx - 1] if vid_idx > 0 else 0)

        out = np.empty((self.num_frames, self.img_size, self.img_size, 3), dtype=np.uint8)
        for i in range(self.num_frames):
            p = frames_list[local_idx + i * self.frameskip]
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img.shape[0] != self.img_size or img.shape[1] != self.img_size:
                img = cv2.resize(img, (self.img_size, self.img_size))
            out[i] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        pixels = torch.from_numpy(out).permute(0, 3, 1, 2).float() / 255.0
        pixels = (pixels - IMAGENET_MEAN) / IMAGENET_STD
        return {"pixels": pixels}
