import bisect
import os

import cv2
import numpy as np
import torch
import torch.utils.data as data

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class SurgicalVideoDataset(data.Dataset):
    """Dataset that loads consecutive frame sequences from local surgery videos.

    Each sample is a dict with key "pixels": (T, C, H, W) float tensor,
    ImageNet-normalized. T = num_frames consecutive frames sampled with frameskip.
    """

    def __init__(self, video_roots, num_frames=4, frameskip=5, img_size=224):
        super().__init__()
        if isinstance(video_roots, str):
            video_roots = [video_roots]
        self.video_roots = video_roots
        self.num_frames = num_frames
        self.frameskip = frameskip
        self.img_size = img_size

        # Raw frames spanned by one clip
        span = (num_frames - 1) * frameskip + 1

        self.videos = []
        self.cumulative_clips = []
        total = 0

        for root in video_roots:
            for vid in sorted(os.listdir(root)):
                path = os.path.join(root, vid)
                cap = cv2.VideoCapture(path)
                n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

                n_clips = max(0, n - span)
                if n_clips > 0:
                    self.videos.append((path, n_clips))
                    total += n_clips
                    self.cumulative_clips.append(total)

        assert total > 0, f"No valid clips found in {video_roots}"
        print(f"SurgicalVideoDataset: {len(self.videos)} videos, {total} clips")

    def __len__(self):
        return self.cumulative_clips[-1]

    def __getitem__(self, idx):
        # To find which video the clip belongs to, this is due to how DataLoader works...
        vid_idx = bisect.bisect_right(self.cumulative_clips, idx)
        path, _ = self.videos[vid_idx]
        local_idx = idx - (self.cumulative_clips[vid_idx - 1] if vid_idx > 0 else 0)

        cap = cv2.VideoCapture(path)
        frames = []
        for i in range(self.num_frames):
            frame_no = local_idx + i * self.frameskip
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
            if not ret:
                frame = frames[-1] if frames else np.zeros(
                    (self.img_size, self.img_size, 3), dtype=np.uint8
                )
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.img_size, self.img_size))
            frames.append(frame)
        cap.release()

        # (T, C, H, W) float [0, 1] then ImageNet normalize, in GSViT this is done in the EfficientViT 
        pixels = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255.0
        pixels = (pixels - IMAGENET_MEAN) / IMAGENET_STD
        return {"pixels": pixels}
