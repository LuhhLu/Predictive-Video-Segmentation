from torchvision.transforms import ToTensor
import os
import shutil
import numpy as np
from tqdm import tqdm
from torchvision.transforms import functional as TF
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
import os
import torch
import torch.nn as nn
import torch.optim as optim
from Unet import Load_unet

to_tensor = transforms.ToTensor()

class unlabel_dataset(Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.videos = [os.path.join(directory, f) for f in os.listdir(directory) if
                       os.path.isdir(os.path.join(directory, f))]
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_path = self.videos[idx]
        frames = []
        if os.path.isdir(video_path):
            image_files = sorted([f for f in os.listdir(video_path) if f.endswith('.png')])
            for img_file in image_files:
                img_path = os.path.join(video_path, img_file)
                try:
                    frame = Image.open(img_path).convert("RGB")
                    frame = self.to_tensor(frame)
                    frames.append(frame)
                except IOError:
                    return video_path, torch.tensor([])
            frames = torch.stack(frames, dim=0)

        return video_path, frames

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"On device: {device}")
    unlabeled_data = unlabel_dataset(directory="Dataset_Student/unlabeled")
    unet_model=Load_unet('best_unet_model_full.pth')
    unet_model.to(device)
    unet_model.eval()
    with torch.no_grad():
        for idx in tqdm(range(0, len(unlabeled_data)), desc="Processing videos"):
            video_path, frames = unlabeled_data[idx]
            if frames.numel() == 0:
                continue
            frames = frames.to(device)
            pred_masks = unet_model(frames)
            pred_masks = torch.sigmoid(pred_masks)
            pred_masks = pred_masks > 0.5
            pred_masks_np = pred_masks.cpu().numpy()
            pred_masks_np = np.argmax(pred_masks_np, axis=1)
            save_path = os.path.join(video_path, 'mask.npy')
            np.save(save_path, pred_masks_np.astype(np.uint8))
    print('finished')
if __name__ == '__main__':
    main()