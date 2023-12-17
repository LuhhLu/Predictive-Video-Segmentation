import os
import shutil
import numpy as np
from tqdm import tqdm
import torch

def prepare_dataset(src_dir, dest_img_dir, dest_mask_dir):
    # Create destination directories if they don't exist
    os.makedirs(dest_img_dir, exist_ok=True)
    os.makedirs(dest_mask_dir, exist_ok=True)

    # Iterate over each video folder
    video_folders = os.listdir(src_dir)
    for video_folder in tqdm(video_folders, desc="Processing videos"):
        video_path = os.path.join(src_dir, video_folder)
        if os.path.isdir(video_path):
            # Copy and rename each image
            image_files = [f for f in os.listdir(video_path) if f.endswith('.png')]
            for img_file in image_files:
                src_img_path = os.path.join(video_path, img_file)
                dest_img_path = os.path.join(dest_img_dir, f'{video_folder}_{img_file}')
                shutil.copy(src_img_path, dest_img_path)

            # Copy and rename the mask file
            mask_path = os.path.join(video_path, 'mask.npy')
            if os.path.exists(mask_path):
                mask = np.load(mask_path)
                for i, frame_mask in enumerate(mask):
                    dest_mask_path = os.path.join(dest_mask_dir, f'{video_folder}_mask_{i}.npy')
                    np.save(dest_mask_path, frame_mask)

# Paths for the source, image, and mask directories
src_train_dir = 'Dataset_Student/train'
dest_train_img_dir = 'unet_train/images'
dest_train_mask_dir = 'unet_train/masks'

src_val_dir = 'Dataset_Student/val'
dest_val_img_dir = 'unet_val/images'
dest_val_mask_dir = 'unet_val/masks'

def main():
    # assume dataset is already downloaded.
    # create two new folder
    # datasets for training and validating Unet
    prepare_dataset(src_train_dir, dest_train_img_dir, dest_train_mask_dir)
    prepare_dataset(src_val_dir, dest_val_img_dir, dest_val_mask_dir)


if __name__ == '__main__':
    main()








