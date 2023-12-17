from torch.utils.data import DataLoader, Subset
import numpy as np
from math import ceil
from load_model_from_ckpt import load_model, get_sampler
from datasets import get_dataset
from runners.ncsn_runner import conditioning_fn
from tqdm import tqdm
import torch
import os
from os.path import expanduser

import argparse
parser = argparse.ArgumentParser(description='Train UNet with custom settings')
parser.add_argument('--val', action='store_true', help='Enable validation (default: enabled)')
parser.add_argument('--no-val', action='store_false', dest='val', help='Disable validation')
parser.set_defaults(val=True)
args = parser.parse_args()



home = expanduser("~")

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

ckpt_path = '../VIT_64/logs/checkpoint.pt'

scorenet, config = load_model(ckpt_path, device)

sampler = get_sampler(config)

if args.val:
    config.data.hidden = False
    dest_file='pred_val.npy'
    dest_path = 'val_paths.txt'
else:
    config.data.hidden = True
    dest_file='pred_hidden.npy'
    dest_path = 'hidden_paths.txt'

_, val_dataset = get_dataset('../Dataset_Student', config, video_frames_pred=0)

# this takes tremendous amount of time, you can subset the hidden file to run on multiple nodes.

val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True,
                        num_workers=64, drop_last=False)

val_y_all = []
pred_all = []

for val_x, val_y in tqdm(val_loader, desc="Processing batches"):
    val_y_all += list(val_y)
    real, cond, cond_mask = conditioning_fn(config, val_x, num_frames_pred=config.sampling.num_frames_pred,
                                            prob_mask_cond=getattr(config.data, 'prob_mask_cond', 0.0),
                                            prob_mask_future=getattr(config.data, 'prob_mask_future', 0.0))
    cond = cond.to(device)
    n_iter_frames = ceil(config.sampling.num_frames_pred / config.data.num_frames)
    init_samples_shape = (real.shape[0], config.data.channels * config.data.num_frames,
                          config.data.image_size, config.data.image_size)
    init = torch.randn(init_samples_shape, device=device)
    pred_samples = []
    for i_frame in range(n_iter_frames):
        gen_samples = sampler(init, scorenet, cond=cond, cond_mask=cond_mask, subsample=1000, verbose=False)
        pred_samples.append(gen_samples)
        if i_frame == n_iter_frames - 1:
            continue
        if cond is None:
            cond = gen_samples
        else:
            cond = torch.cat([cond[:, config.data.channels * config.data.num_frames:],
                              gen_samples[:,
                              config.data.channels * max(0, config.data.num_frames - config.data.num_frames_cond):]],
                             dim=1)
    pred = torch.cat(pred_samples, dim=1)[:, :config.data.channels * config.sampling.num_frames_pred]
    pred_all.append(pred.cpu().numpy().reshape(pred.shape[0], config.sampling.num_frames_pred, config.data.channels,
                                               config.data.image_size, config.data.image_size))
pred_all_np = np.concatenate(pred_all, axis=0)


# if you subset the hidden set, mind to change here also

# np.save(f'../pred_all_1.npy', pred_all_np)
#
# with open(f'../val_y_paths_1.txt', 'w') as file:
#     for path in val_y_all:
#         file.write(path + '\n')

# 保存 pred_all_np
np.save(os.path.join('..', dest_file), pred_all_np)

# 写入路径到文件
try:
    with open(os.path.join('..', dest_path), 'w') as file:
        for path in val_y_all:
            file.write(path + '\n')
except IOError as e:
    print(f"Error writing to file: {e}")
