import os
import random
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data

# Avoid thread oversubscription: the async pipeline already uses a thread pool.
cv2.setNumThreads(0)

FRAME_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _load_clip(frame_paths, start, clip_dur, img_size):
    """Load a consecutive clip of `clip_dur` frames starting at index `start`."""
    clip = np.empty((clip_dur, img_size, img_size, 3), dtype=np.uint8)
    for i in range(clip_dur):
        img = cv2.imread(frame_paths[start + i], cv2.IMREAD_COLOR)
        if img.shape[0] != img_size or img.shape[1] != img_size:
            img = cv2.resize(img, (img_size, img_size))
        clip[i] = img  # BGR (matches original behavior — downstream didn't convert)
    return clip


class SurgicalDataset(data.Dataset):
    """Samples (input, target) frame pairs from pre-extracted JPEG frames.

    Expected root layout (produced by le-wm/extract_frames.py):
        root/
            video_a/frame_000001.jpg, frame_000002.jpg, ...
            video_b/...

    Preserves the async-pipeline API expected by pretrain_model.py:
        parallel_generate(), generate_dataset(parallel_call=True),
        get(idx), total_frames.
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
                 loader_threads=16):
        super().__init__()

        self.root = root
        self.is_train = is_train
        self.finetune = finetune
        self.batch_size = batch_size
        self.predict_change = predict_change
        self.transform = transform
        self.img_size = img_size

        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.n_frames_total = n_frames_input + n_frames_output
        self.clip_dur = self.n_frames_total

        self.videos = [] 
        total = 0
        for name in sorted(os.listdir(root)):
            vdir = os.path.join(root, name)
            if not os.path.isdir(vdir):
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

        # Two thread pools: one for individual clip loads, one that owns the async fill.
        self._inner = ThreadPoolExecutor(max_workers=loader_threads)
        self._outer = ThreadPoolExecutor(max_workers=1)
        self._next_future = None

        self.clips = None  # uint8 CPU tensor (batch_size, clip_dur, H, W, 3)

        self.std = 1
        self.mean = 0

    def _sample_one_clip(self):
        vi = int(np.random.choice(len(self.videos), p=self.video_probs))
        _, frames = self.videos[vi]
        start = random.randint(0, len(frames) - self.clip_dur)
        return _load_clip(frames, start, self.clip_dur, self.img_size)

    def _fill_clips(self):
        buf = np.empty(
            (self.batch_size, self.clip_dur, self.img_size, self.img_size, 3),
            dtype=np.uint8,
        )
        futures = [self._inner.submit(self._sample_one_clip) for _ in range(self.batch_size)]
        for i, f in enumerate(futures):
            buf[i] = f.result()
        return torch.from_numpy(buf)

    def parallel_generate(self):
        """Prime the pipeline: fill the current buffer and queue the next one."""
        if self.clips is None:
            self.clips = self._fill_clips()
        if self._next_future is None:
            self._next_future = self._outer.submit(self._fill_clips)

    def generate_dataset(self, parallel_call=False):
        """Swap in the background-filled buffer and queue the next refill."""
        if parallel_call:
            if self._next_future is None:
                self.parallel_generate()
                return
            self.clips = self._next_future.result()
            self._next_future = self._outer.submit(self._fill_clips)
        else:
            self.clips = self._fill_clips()

    def __len__(self):
        return self.total_frames

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
    )


def finetune_data(num_images, data_root, num_workers, predict_change=False):
    return SurgicalDataset(
        root=data_root, is_train=True, batch_size=num_images, finetune=True,
        n_frames_input=1, n_frames_output=1, predict_change=predict_change,
    )


if __name__ == "__main__":
    import sys
    ds = load_data(num_images=16, data_root=sys.argv[1], num_workers=1, gpu=False)
    ds.parallel_generate()
    ds.generate_dataset(parallel_call=True)
    inp, out = ds.get(torch.arange(4))
    print(inp.shape, out.shape, inp.dtype, inp.min().item(), inp.max().item())
