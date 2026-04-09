import os
import gzip
import pickle
import random
import time
import torch.nn.functional as F

import numpy as np
import torch
import torch.utils.data as data
from torch.multiprocessing import Process
import torch.multiprocessing as mp #, Queue


from numpy import *
import cv2
from PIL import Image

def read_video(n_frames=None, video_loc=None):
    i = 0
    all = []
    cap = cv2.VideoCapture(video_loc) #"rec_q26b_10min.mp4")
    if n_frames is None:
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while cap.isOpened() and i < n_frames:
        ret, frame = cap.read()
        arr = np.array(frame)
        all.append(arr)
        i += 1
    return np.array(all)

def load_surgical(root, frames=None):
    videos = list()
    for video in os.listdir(root):
        videos.append(read_video(video_loc=root + "/" + video, n_frames=frames))
    return videos


def lazy_load_surgical(arr, root, gpu):
    clips = list()
    videos= os.listdir(root)
    num_videos=100
    clip_dur=2

    sampled_videos = [random.choice(videos,) for _ in range(num_videos)]
    for video in sampled_videos:
        cap = cv2.VideoCapture(root + video)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_frame = random.randint(0, n_frames - 1 - clip_dur)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        all_frames = []
        for _ in range(clip_dur):  # Only read the required number of frames
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.resize(frame, (224, 224))
            all_frames.append(np.array(frame))
        cap.release()
        clip = np.array(all_frames)
        clip_tensor = torch.tensor(clip, device="cuda:3" if gpu else "cpu").unsqueeze(0)
        # print(clip_tensor.size())
        clips.append(clip_tensor)
    

    if arr is not None: 
        arr[:] = torch.cat(clips, dim=0).squeeze()[:].clone()
    else:
        return clips


class SurgicalDataset(data.Dataset):
    def __init__(self, 
                 root, 
                 is_train=True, 
                 n_frames_input=1, 
                 n_frames_output=1, 
                 transform=None,
                 batch_size=128,
                 predict_change=False, 
                 gpu=True,
                 finetune=False):
        super(SurgicalDataset, self).__init__()

        self.root = root
        self.dataset = None
        self.gpu = gpu
        self.finetune = finetune
        self.batch_size = batch_size
        self.predict_change = predict_change
        self.videos = os.listdir(root)
        self.num_video = len(self.videos)

        total_frames = 0
        video_probs = list()
        for video in self.videos:
            cap = cv2.VideoCapture(root + video)
            print(root + video)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            total_frames += frames
            video_probs.append(frames)
        for _ in range(len(video_probs)):
            video_probs[_] /= total_frames
        self.video_probs = video_probs
        self.total_frames = total_frames

        self.is_train = is_train
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.n_frames_total = self.n_frames_input + self.n_frames_output
        self.transform = transform

        # data format:
        # (time in + time out) X batch X image_x X image_y x 1
        self.clips = list()
        self.generate_dataset()

        self.std = 1
        self.mean = 0
        mp.set_start_method('spawn')

        self.procs = 12
        self.proc_id = 0
        for proc in range(self.procs):
            setattr(self, "parallel_proc{}".format(proc), None)
            setattr(self, "return_arr{}".format(proc),
                    torch.zeros((batch_size, 2, 224, 224, 3), device="cuda:3" if self.gpu else "cpu").share_memory_())

    def parallel_generate(self, proc_id=None):
        if proc_id is not None:
            setattr(self, "parallel_proc{}".format(proc_id), Process(
                target=lazy_load_surgical,
                args=(getattr(self, "return_arr{}".format(proc_id)),self.root, self.gpu)))
            getattr(self, "parallel_proc{}".format(proc_id)).start()
        else:
            for proc in range(self.procs):
                setattr(self, "parallel_proc{}".format(proc), Process(
                    target=lazy_load_surgical,
                    args=(getattr(self, "return_arr{}".format(proc)), self.root, self.gpu)))
                getattr(self, "parallel_proc{}".format(proc)).start()


    def generate_dataset(self, parallel_call=False):
        """
        We want to take random clip segments from videos
        todo: randomize the video "speed" interpolating between frames?
        """
        if parallel_call:
            # wait for process to return dataset
            getattr(self, "parallel_proc{}".format(self.proc_id)).join()
            # get return array from parallel process
            self.clips = getattr(self, "return_arr{}".format(self.proc_id)).clone()
            # calculate length
            self.num_clips= self.clips.shape[0]
            # regenerate process
            self.parallel_generate(proc_id=self.proc_id)
            # set pointer to next process
            self.proc_id = (self.proc_id + 1) % self.procs
            return
        lazy_load_dataset = lazy_load_surgical(root=self.root, arr=None, gpu=self.gpu)
        self.clips = lazy_load_dataset
        self.clips = [torch.tensor(_, device="cuda:3" if self.gpu else "cpu").unsqueeze(0).clone().detach() for _ in self.clips]
        self.clips = torch.cat(self.clips, dim=0)
        self.num_clips = len(self.clips)

    def __len__(self):
        return self.total_frames

    def get(self, idx):
        # Define the resize transformation
        clips = self.clips[idx].squeeze()
        inp = (clips[:, 0:1, :, :, :] / 255.0).contiguous().float().squeeze().permute(0, 2, 3, 1).permute(0, 2, 3, 1)
        out = (clips[:, 1:2, :, :, :] / 255.0).contiguous().float().squeeze().permute(0, 2, 3, 1).permute(0, 2, 3, 1)
        inp = F.interpolate(inp, size=(224, 224), mode='bilinear', align_corners=False)
        out = F.interpolate(out, size=(224, 224), mode='bilinear', align_corners=False)
        if self.predict_change:
            out = out - inp
        return inp, out


def load_data(num_images, data_root, num_workers, predict_change=False, gpu=True):
    train_set = SurgicalDataset(
        root=data_root, is_train=True, batch_size=num_images,
        n_frames_input=1, n_frames_output=1, predict_change=predict_change, gpu=gpu)
    return train_set

def finetune_data(num_images, data_root, num_workers, predict_change=False):
    train_set = SurgicalDataset(
        root=data_root, is_train=True, batch_size=num_images, finetune=True,
        n_frames_input=1, n_frames_output=1, predict_change=predict_change)
    return train_set

if __name__ == "__main__":
    dataloader_train = load_data(10000, 1, "./data/", 1)
    import nvsmi
    print(nvsmi.get_gpu_processes())
