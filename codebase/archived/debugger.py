import torch
import segmentation_models_pytorch as smp
import monai

from data_organization import HIE_Dataset, reassemble_to_3d
from pipeline_utils import *
from transforms.preprocess_v3 import transform_2d, padding
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np


DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
print(DEVICE)
# ENCODER = "se_resnext101_32x4d"
ENCODER = "densenet161"
DATA_ROOT = "/Users/amograo/Desktop/DATASET"

metrics_3d = [
    ('Dice',monai.metrics.DiceMetric(include_background=True,ignore_empty=False),0.0),
    ('MASD',monai.metrics.SurfaceDistanceMetric(include_background=False, symmetric = True),np.inf),
    ('NSD',monai.metrics.SurfaceDiceMetric(include_background=False, distance_metric="euclidean", class_thresholds=[2]),0.0)
]
metrics = [
            ('Dice', monai.metrics.DiceMetric(include_background=True,ignore_empty=False)),  
            ('IoU', monai.metrics.MeanIoU(include_background=True,ignore_empty=False)), 
        ]

loss = monai.losses.DiceFocalLoss()

model = smp.Unet(
    encoder_name=ENCODER,
    encoder_weights=None,     
    in_channels=2,             
    classes=1,
    activation='sigmoid'               
)

model.to(DEVICE)

# checkpoint = torch.load("/home/lakes/bonbid-2024/checkpoints/2D_densenet161-Aug-Stacked/models/latest_model_densenet161.pth",map_location=torch.device(DEVICE))  
checkpoint = torch.load("/Users/amograo/Downloads/model_epoch_180_densenet161_3d.pth",map_location=torch.device(DEVICE))  
model.load_state_dict(checkpoint['model_state_dict'])

df = pd.read_csv(f'{DATA_ROOT}/BONBID2024_Val/metadata.csv')
uids = [str(i).zfill(3) for i in df["Patient ID"].tolist()]
image_paths = [f'{DATA_ROOT}/BONBID2024_Val/ADC',f'{DATA_ROOT}/BONBID2024_Val/Z_ADC'] #f'{DATA_ROOT}/BONBID2024_Val/ADC',

val_dataset = HIE_Dataset(
    images_dir = [f'{DATA_ROOT}/BONBID2024_Val/ADC',f'{DATA_ROOT}/BONBID2024_Val/Z_ADC'],
    masks_dir = f'{DATA_ROOT}/BONBID2024_Val/LABEL',
    csv_file = f'{DATA_ROOT}/BONBID2024_Val/metadata.csv',
    dimension = '2D',
    transform=transform_2d
)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# valid_logs = epoch_runner("val", val_loader, model, loss, metrics, device=DEVICE)
inference_3d_runner(image_paths, f'{DATA_ROOT}/BONBID2024_Val/LABEL', uids, ['ADC','Z_ADC'], model,metrics_3d, DEVICE)
valid_logs = epoch_runner("val", val_loader, model, loss, metrics, device=DEVICE)
inference_3d_runner(image_paths, f'{DATA_ROOT}/BONBID2024_Val/LABEL', uids, ['ADC','Z_ADC'], model,metrics_3d, DEVICE)
# inference_3d_runner(image_paths, f'{DATA_ROOT}/BONBID2024_Val/LABEL', uids, ['ADC','Z_ADC'], model,metrics_3d, DEVICE)

