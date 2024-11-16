# should pick up the model based on the path
# should load state dict
# should make predictions and save images => GT, PR, and overlay
# should mention the dice overlap as per MONAI
# read images from the dataset class

# %%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from torchvision import transforms

import segmentation_models_pytorch as smp
import monai

from data_organization import HIE_Dataset
from pipeline_utils import *
from transforms.preprocess import resample

import os
import wandb

DATA_ROOT = ""
MODEL_DIR = ""
EPOCH = 0
ENCODER = "densenet121"
BATCH_SIZE = 16

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

# %%
model = smp.Unet(
    encoder_name=ENCODER,
    encoder_weights=None,     
    in_channels=1,             
    classes=1,
    activation='sigmoid'               
)

model.to(DEVICE)

_,_,_ = resume_checkpoint(MODEL_DIR, model, None, DEVICE, epoch=EPOCH, string=f"_{ENCODER}")

loss = monai.losses.DiceLoss(sigmoid=False)

metrics = [
            ('Dice', monai.metrics.DiceMetric(include_background=True,ignore_empty=False)),  
        ]

val_dataset = HIE_Dataset(
    images_dir = [f'{DATA_ROOT}/BONBID2023_Val/2Z_ADC'],
    masks_dir = f'{DATA_ROOT}/BONBID2023_Val/3Label',
    csv_file = f'{DATA_ROOT}/BONBID2023_Val/df.csv',
    mode = '2d',
    transform=resample
)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

for i in range(1):
    epoch_metrics = epoch_runner_save_image(
        "val",
        val_loader,
        model,
        loss,
        metrics,
        device=DEVICE,
        save_images=True,
        save_path="saved_images"
    )