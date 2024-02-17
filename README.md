# Video Segmentation

This Project is for Predicting video segmentation of 22nd frame condition on first 11 frames. Result Slides: https://www.figma.com/proto/DQLuQ3slIPkKiGM4axxDFD/FProject?node-id=1-3&t=dN3eV11w9K2J3QLq-1&mode=design

## Getting Started

These instructions will guide you through the setup and execution of the project.

### Prerequisites

Prerequisites and dependencies that need to be installed:

```
pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
numpy
PyYAML
imageio
imageio-ffmpeg
matplotlib
opencv-python
scikit-image
tqdm
h5py
progressbar
psutil
ninja
gdown
zipfile36
argparse
torchmetrics
```

you can also find them in ` requirements.txt `

### Installation

A step-by-step series of examples that tell you how to get a development environment running.

1. **Download and Unzip Dataset:**

   Use the provided script to download and unzip the dataset.(Estimate 1hr, 24 CPU)

   ```bash
   python data_download.py
   ```

2. **Prepare Datasets for Training U-Net:**

   Prepare the necessary datasets for training the U-Net models. (Estimate 10 min, 24 CPU)

   ```bash
   python data_preparation.py
   ```

### Training U-Nets

Train U-Net models with different resolutions. It's required to train both 'full' and '64' resolutions to run experiments. Other resolutions can be explored.

1. **Training U-Net with Full Resolution:**
   (Estimate 15min, 4 V100 GPUs)
   ```bash
   python unet_train.py --res full --lr 0.001 --batch 64 --epoch 10
   ```

2. **Training U-Net with 64x64 Resolution:**
   (Estimate 15min, 4 V100 GPUs)
   ```bash
   python unet_train.py --res 64 --lr 0.001 --batch 64 --epoch 10
   ```

### Post-Training Procedures

After training the U-Nets, follow these steps:

1. **Generate Masks for Unlabeled Dataset:** (Estimate 10 hours)

   ```bash
   python unlabel_masks.py
   ```

2. **Generate Samples for Each Class: (optional)** (Estimate 5 mins)

   ```bash
   python samples.py
   ```

3. **Generate Fine-tune Dataset:** (Estimate 5 mins)

   ```bash
   python video_screen.py
   ```

### Pre-training Video-Prediction Model

Pre-train the video-prediction model using the following command: (Estimate 1 day)

```bash
cd mcvd-pytorch
python main.py --config configs/VIT.yml --data_path ../Dataset_Student --exp ../VIT_64
```

### Fine-tuning Video-Prediction Model

Fine-tune the video-prediction model with the modified configuration:

```bash
cd mcvd-pytorch
python main.py --config configs/VIT.yml --data_path ../Dataset_Student --exp ../VIT_64 --resume_training --config_mod data.finetune=True
```

### Generating Images for Hidden and Validation Sets

1. **For Hidden Set:** (Estimate 1 day)

   ```bash
   cd mcvd-pytorch
   python prediction.py --no-val
   ```

2. **For Validation Set:** (Optional) (Estimate 12 hours)

   ```bash
   cd mcvd-pytorch
   python prediction.py --val
   ```

### Predicting for Hidden Set

Final results are stored at 'hidden_22_pred.npy'. (Estimate 2 hours)

   ```bash
   python mask_pred.py --no-val
   ```


### Testing on Validation Set

Test the model on the validation set: (Estimate 1 hour)

   ```bash
   python mask_pred.py --val
   ```

---
