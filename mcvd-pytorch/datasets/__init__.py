import numpy as np
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from datasets.VIT import VIT_dataset
from torch.utils.data import Subset


DATASETS = ['VIT_DATASET']


def get_dataloaders(data_path, config):
    dataset, test_dataset = get_dataset(data_path, config)
    dataloader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=True,
                            num_workers=config.data.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=True,
                             num_workers=config.data.num_workers, drop_last=True)
    return dataloader, test_loader


def get_dataset(data_path, config, video_frames_pred=0, start_at=0):

    assert config.data.dataset.upper() in DATASETS, \
        f"datasets/__init__.py: dataset can only be in {DATASETS}! Given {config.data.dataset.upper()}"

    if config.data.random_flip is False:
        tran_transform = test_transform = transforms.Compose([
            transforms.Resize(config.data.image_size),
            transforms.ToTensor()
        ])
    else:
        tran_transform = transforms.Compose([
            transforms.Resize(config.data.image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()
        ])
        test_transform = transforms.Compose([
            transforms.Resize(config.data.image_size),
            transforms.ToTensor()
        ])

    if config.data.dataset.upper() == "VIT_DATASET":
        if getattr(config.data, "finetune", 0):
            train_path='finetune_set'
        else:
            train_path = 'unlabeled'
        train_directory = os.path.join(data_path, train_path)
        frames_per_sample = config.data.num_frames_cond + getattr(config.data, "num_frames_future",
                                                                  0) + video_frames_pred
        frame_resize = getattr(config.data, "frame_resize", 0)
        dataset = VIT_dataset(train_directory, frame_resize, frame_length=frames_per_sample,
                              transform_flip=getattr(config.data, "transform_flip", 0))
        print('train set:', len(dataset))


        if getattr(config.data, "hidden", 0):
            val_path='hidden'
        else:
            val_path = 'val'
        test_directory = os.path.join(data_path, val_path)
        print(test_directory)
        test_dataset = VIT_dataset(test_directory, frame_resize, frame_length=frames_per_sample,
                                   transform_flip=getattr(config.data, "transform_flip", 0))
        print('test set:', len(test_dataset))
    else:
        raise ValueError('Wrong dataset specified')

    subset_num = getattr(config.data, "subset", -1)
    if subset_num > 0:
        subset_indices = list(range(subset_num))
        dataset = Subset(dataset, subset_indices)

    test_subset_num = getattr(config.data, "test_subset", -1)
    if test_subset_num > 0:
        subset_indices = list(range(test_subset_num))
        test_dataset = Subset(test_dataset, subset_indices)

    return dataset, test_dataset


def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)


def data_transform(config, X):
    if config.data.uniform_dequantization:
        X = X / 256. * 255. + torch.rand_like(X) / 256.
    if config.data.gaussian_dequantization:
        X = X + torch.randn_like(X) * 0.01

    if config.data.rescaled:
        X = 2 * X - 1.
    elif config.data.logit_transform:
        X = logit_transform(X)

    if hasattr(config, 'image_mean'):
        return X - config.image_mean.to(X.device)[None, ...]

    return X


def inverse_data_transform(config, X):
    if hasattr(config, 'image_mean'):
        X = X + config.image_mean.to(X.device)[None, ...]

    if config.data.logit_transform:
        X = torch.sigmoid(X)
    elif config.data.rescaled:
        X = (X + 1.) / 2.

    return torch.clamp(X, 0.0, 1.0)
