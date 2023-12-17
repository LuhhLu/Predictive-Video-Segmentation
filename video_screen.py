import numpy as np
import os
import shutil
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

def Anomalies_drop(mask):
    unique_numbers=np.unique(mask)
    count_dict = {num: np.count_nonzero(mask == num) for num in unique_numbers}
    anomalies = [key for key, value in count_dict.items() if value < 50]
    for anomaly in anomalies:
        mask[mask == anomaly] = 0
    return mask

def process_video_folder(video_folder, src_dir, dest_dir, is_train=True):
    video_path = os.path.join(src_dir, video_folder)
    if os.path.isdir(video_path):
        mask_file = os.path.join(video_path, 'mask.npy')
        if os.path.exists(mask_file):
            mask = np.load(mask_file)
            if is_train:
                mask = Anomalies_drop(mask)
            prev11 = np.unique(mask[0:11])
            post11 = np.unique(mask[-11:])
            if np.array_equal(prev11, post11):
                dest_folder_path = os.path.join(dest_dir, video_folder)
                shutil.copytree(video_path, dest_folder_path)

def prepare_consistent_dataset(src_dir_train, src_dir_unlabel, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)

    video_folders_train = os.listdir(src_dir_train)
    video_folders_unlabel = os.listdir(src_dir_unlabel)

    with Pool() as pool:
        pool.starmap(process_video_folder, [(folder, src_dir_unlabel, dest_dir, False) for folder in video_folders_unlabel])
        pool.starmap(process_video_folder, [(folder, src_dir_train, dest_dir, True) for folder in video_folders_train])


def main():
    src_dir_train = 'Dataset_Student/train'
    src_dir_unlabel = 'Dataset_Student/unlabeled'
    dest_dir = '/scratch/hl5438/finetune_set'

    prepare_consistent_dataset(src_dir_train, src_dir_unlabel, dest_dir)

    file_count = len(os.listdir('/scratch/hl5438/pred_trainset'))
    print(f"Total number of finetune files in '{dest_dir}': {file_count}")

if __name__ == '__main__':
    main()
