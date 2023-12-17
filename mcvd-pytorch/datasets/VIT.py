import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random


class VIT_dataset(Dataset):
    def __init__(self, directory, frame_resize, frame_length, transform_flip, with_target=True):
        """
        Args:
            directory (string)
            config
        """
        self.directory = directory
        parts = directory.split('/')
        if parts[-1] == 'hidden':
            self.videos = [os.path.join(self.directory, f)
                           for f in os.listdir(self.directory)
                           if os.path.isdir(os.path.join(self.directory, f))]
            self.frame_length = 11
            self.extra_frame = 0

        else:
            self.videos = [os.path.join(self.directory, f)
                           for f in os.listdir(self.directory)
                           if os.path.isdir(os.path.join(self.directory, f)) and
                           os.path.exists(os.path.join(self.directory, f, 'mask.npy'))]
            self.frame_length = frame_length
            if self.frame_length > 22:
                raise ValueError(
                    "The total number of frames (num_frames_cond + num_frames + num_frames_future) exceeds the limit of 22.")
            else:
                self.extra_frame = 22 - self.frame_length

        self.frame_resize = frame_resize
        self.transform_flip = transform_flip
        self.with_target = with_target
        self.p = 0.5

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_path = self.videos[idx]
        images = [f for f in os.listdir(video_path) if f.endswith('.png')]
        images.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        # Transform to apply to each frame
        transform = transforms.Compose([
            transforms.Resize((self.frame_resize, self.frame_resize)),
            transforms.ToTensor(),
        ])
        frames = []
        for image_file in images:
            image_path = os.path.join(video_path, image_file)
            image = Image.open(image_path)
            image_tensor = transform(image)
            frames.append(image_tensor)
        frames = torch.stack(frames, dim=0)
        start_frame = random.randint(0, self.extra_frame)
        frames = frames[start_frame:start_frame + self.frame_length]
        # Transformation
        if self.transform_flip:
            if torch.rand(1) < self.p:
                frames = torch.flip(frames, [3])
        if self.with_target:
            return frames, video_path
        else:
            return frames