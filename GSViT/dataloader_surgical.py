import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data

# Each DataLoader worker is its own process; keep OpenCV single-threaded.
if hasattr(cv2, "setNumThreads"):
    cv2.setNumThreads(0)

FRAME_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _load_clip(frame_paths, start, clip_dur, img_size):
    """Load a consecutive clip of `clip_dur` frames starting at index `start`."""
    clip = np.empty((clip_dur, img_size, img_size, 3), dtype=np.uint8)
    for i in range(clip_dur):
        img = cv2.imread(frame_paths[start + i], cv2.IMREAD_COLOR)
        if img.shape[0] != img_size or img.shape[1] != img_size:
            img = cv2.resize(img, (img_size, img_size))
        clip[i] = img  # BGR (matches original GSViT behavior)
    return clip


class SurgicalDataset(data.Dataset):
    """Samples (input, target) frame pairs from pre-extracted JPEG frames.

    Expected root layout (produced by le-wm/extract_frames.py):
        root/
            video_a/frame_000001.jpg, frame_000002.jpg, ...
            video_b/...

    Preserves the async-pipeline API expected by pretrain_model.py:
        parallel_generate(), generate_dataset(parallel_call=True),
        get(idx), total_frames, self.clips.

    Internally backed by a torch.utils.data.DataLoader with persistent workers
    and prefetch, so several batches are always in flight.
    """

    def __init__(self,
                 root,
                 is_train=True,
                 n_frames_input=1,
                 n_frames_output=1,
                 transform=None,
                 batch_size=128,
                 predict_change=False,
                 gpu=True,
                 finetune=False,
                 img_size=224,
                 num_workers=12,
                 prefetch_factor=4):
        super().__init__()

        self.root = root
        self.is_train = is_train
        self.finetune = finetune
        self.batch_size = batch_size
        self.predict_change = predict_change
        self.transform = transform
        self.img_size = img_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.n_frames_total = n_frames_input + n_frames_output
        self.clip_dur = self.n_frames_total

        self.videos = []
        total = 0
        for name in sorted(os.listdir(root)):
            vdir = os.path.join(root, name)
            if not (os.path.isdir(vdir) or os.path.islink(vdir)):
                continue
            frames = sorted(
                os.path.join(vdir, f) for f in os.listdir(vdir)
                if os.path.splitext(f)[1].lower() in FRAME_EXTS
            )
            if len(frames) >= self.clip_dur:
                self.videos.append((name, frames))
                total += len(frames)

        assert self.videos, f"No frame folders with >= {self.clip_dur} frames under {root}"
        self.total_frames = total

        counts = np.array([len(f) for _, f in self.videos], dtype=np.float64)
        self.video_probs = counts / counts.sum()

        self._loader = None
        self._iter = None
        self.clips = None  # uint8 CPU tensor (batch_size, clip_dur, H, W, 3)

        self.std = 1
        self.mean = 0

    # --- Dataset protocol: idx is ignored; sampling is random-weighted by video length. ---
    def __len__(self):
        return self.total_frames

    def __getitem__(self, idx):
        vi = int(np.random.choice(len(self.videos), p=self.video_probs))
        _, frames = self.videos[vi]
        start = random.randint(0, len(frames) - self.clip_dur)
        clip = _load_clip(frames, start, self.clip_dur, self.img_size)
        return torch.from_numpy(clip)

    # --- Async pipeline API (DataLoader-backed). ---
    def _ensure_loader(self):
        if self._loader is not None:
            return
        kwargs = dict(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,   # randomness lives inside __getitem__
            drop_last=True,
            pin_memory=False,
        )
        if self.num_workers > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = self.prefetch_factor
        self._loader = data.DataLoader(self, **kwargs)
        self._iter = iter(self._loader)

    def _next_batch(self):
        try:
            return next(self._iter)
        except StopIteration:
            self._iter = iter(self._loader)
            return next(self._iter)

    def parallel_generate(self):
        """Prime the pipeline: fetch the first batch (workers keep prefetching behind it)."""
        self._ensure_loader()
        if self.clips is None:
            self.clips = self._next_batch()

    def generate_dataset(self, parallel_call=False):
        """Advance to the next prefetched batch."""
        self._ensure_loader()
        self.clips = self._next_batch()

    def get(self, idx):
        """Return (inp, out) batch tensors of shape (n, C, H, W), float in [0, 1].

        inp = frame t, out = frame t+1 (or diff if predict_change).
        `idx`: LongTensor of indices into self.clips (length <= batch_size).
        """
        clips = self.clips[idx]  # (n, clip_dur, H, W, 3) uint8, CPU
        inp = clips[:, 0].permute(0, 3, 1, 2).float() / 255.0
        out = clips[:, 1].permute(0, 3, 1, 2).float() / 255.0
        if inp.shape[-1] != self.img_size or inp.shape[-2] != self.img_size:
            inp = F.interpolate(inp, size=(self.img_size, self.img_size),
                                mode="bilinear", align_corners=False)
            out = F.interpolate(out, size=(self.img_size, self.img_size),
                                mode="bilinear", align_corners=False)
        if self.predict_change:
            out = out - inp
        return inp, out


def load_data(num_images, data_root, num_workers, predict_change=False, gpu=True):
    return SurgicalDataset(
        root=data_root, is_train=True, batch_size=num_images,
        n_frames_input=1, n_frames_output=1,
        predict_change=predict_change, gpu=gpu,
        num_workers=num_workers,
    )


def finetune_data(num_images, data_root, num_workers, predict_change=False):
    return SurgicalDataset(
        root=data_root, is_train=True, batch_size=num_images, finetune=True,
        n_frames_input=1, n_frames_output=1, predict_change=predict_change,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    import sys
    ds = load_data(num_images=16, data_root=sys.argv[1], num_workers=2, gpu=False)
    ds.parallel_generate()
    ds.generate_dataset(parallel_call=True)
    inp, out = ds.get(torch.arange(4))
    print(inp.shape, out.shape, inp.dtype, inp.min().item(), inp.max().item())
