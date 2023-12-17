import torch
import os
import numpy as np
from PIL import Image
import random
import torch.nn as nn
from torch.utils.data import Dataset
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Downscaling paths
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.MaxPool2d(2)
        self.down1_conv = DoubleConv(64, 128)
        self.down2 = nn.MaxPool2d(2)
        self.down2_conv = DoubleConv(128, 256)
        self.down3 = nn.MaxPool2d(2)
        self.down3_conv = DoubleConv(256, 512)
        self.down4 = nn.MaxPool2d(2)
        self.down4_conv = DoubleConv(512, 1024)  # Increased to match the channels

        # Upscaling paths
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv1 = DoubleConv(1024, 512)  # Corrected channel size
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv4 = DoubleConv(128, 64)

        # Output layer
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2 = self.down1_conv(x2)
        x3 = self.down2(x2)
        x3 = self.down2_conv(x3)
        x4 = self.down3(x3)
        x4 = self.down3_conv(x4)
        x5 = self.down4(x4)
        x5 = self.down4_conv(x5)

        # Decoder path
        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.up_conv1(x)
        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.up_conv2(x)
        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up_conv3(x)
        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up_conv4(x)

        logits = self.outc(x)
        return logits


class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, weights, reduction='mean'):
        super().__init__()
        self.weights = weights
        self.reduction = reduction
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce_loss(inputs, targets)

        weights_expanded = self.weights.view(1, -1, 1, 1).to(bce_loss.device)

        weighted_loss = bce_loss * weights_expanded

        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss

def Load_unet(path=None):
    if path:
        unet_model = UNet(n_channels=3, n_classes=49)
        unet_model.load_state_dict(torch.load(path))
    else:
        unet_model = UNet(n_channels=3, n_classes=49)
    return unet_model


def one_hot_encode_mask(mask, num_classes):
    one_hot = np.zeros((num_classes, mask.shape[0], mask.shape[1]), dtype=np.float32)
    for c in range(num_classes):
        one_hot[c, :, :] = (mask == c)
    return one_hot



class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Update mask name to match new naming convention and extension
        mask_name = img_name.replace('image', 'mask').replace('.png', '.npy')
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(img_path).convert("RGB")
        mask = np.load(mask_path)  # Load mask as numpy array

        if self.transform:
            seed = np.random.randint(2147483647)  # Random seed for consistent transformations
            random.seed(seed)
            torch.manual_seed(seed)

            # Apply transformations to the image
            image = self.transform(image)

            # One-hot encode the transformed mask
            mask = one_hot_encode_mask(mask, 49)

            # Convert the mask back to a tensor
            mask = torch.from_numpy(mask)

        return image, mask