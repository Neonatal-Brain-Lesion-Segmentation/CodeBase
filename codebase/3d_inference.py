import torch
import segmentation_models_pytorch as smp
import monai

from data_organization import HIE_Dataset, reassemble_to_3d
from pipeline_utils import *
from transforms.preprocess import resample

import pandas as pd
import numpy as np

def crop_to_original(data: np.ndarray, original_size: tuple = (144, 160)) -> np.ndarray:
    """
    Crops 2D slices (numpy arrays) from a target size back to the original size.
    
    Parameters:
    - data (np.ndarray): Input array of shape (D, H, W).
    - original_size (tuple): The target size to crop back to (Height, Width).
    
    Returns:
    - np.ndarray: The cropped array of shape (D, original_size[0], original_size[1]).
    """
    D, H, W = data.shape
    target_height, target_width = original_size

    # Calculate crop indices
    start_h = (H - target_height) // 2
    end_h = start_h + target_height
    start_w = (W - target_width) // 2
    end_w = start_w + target_width

    # Perform cropping
    cropped_data = data[:, start_h:end_h, start_w:end_w]
    return cropped_data

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
print(DEVICE)
ENCODER = "se_resnext101_32x4d"
DATA_ROOT = "/Users/amograo/Desktop/DATASET"

model = smp.Unet(
    encoder_name=ENCODER,
    encoder_weights=None,     
    in_channels=1,             
    classes=1,
    activation='sigmoid'               
)

model.to(DEVICE)

checkpoint = torch.load("/Users/amograo/Research_Projects/DL_HIE_2024/model_epoch_67_se_resnext101_32x4d.pth",map_location=torch.device(DEVICE))    
model.load_state_dict(checkpoint['model_state_dict'])


df = pd.read_csv(f'{DATA_ROOT}/BONBID2024_Val/metadata.csv')
uids = [str(i).zfill(3) for i in df["Patient ID"].tolist()]
print(uids)

preds_3d = {i: [] for i in uids}
masks_3d = {uid: reassemble_to_3d(f'{DATA_ROOT}/BONBID2024_Val/LABEL', uid) for uid in uids}

# mask = np.expand_dims(masks_3d["436"][10],axis=0)
########
# mask = np.array(masks_3d["436"])
# print(mask.shape)
# print(resample(mask).shape)
# print(crop_to_original(resample(mask)).shape)

# print(np.array_equal(mask, crop_to_original(resample(mask))))
#########

#######

dice_l = []
masd_l = []
nsd_l = []
for uid in uids:
    image_set = reassemble_to_3d(f'{DATA_ROOT}/BONBID2024_Val/Z_ADC', uid)

    for i in range(image_set.shape[0]):
        image = np.expand_dims(resample(np.stack([image_set[i]])),axis=0)

        image = torch.tensor(image).to(DEVICE)

        output = model(image)
        pred = (output >= 0.5).float()


        preds_3d[uid].append(crop_to_original(pred.cpu().detach().numpy()[0],original_size=tuple(image_set.shape[1:]))[0])

        # print(pred.shape)
        # print(np.array(preds_3d[uid]).shape)
        if len(preds_3d[uid]) == image_set.shape[0]:
            preds_3d[uid] = np.stack(preds_3d[uid])
            dice = monai.metrics.DiceMetric(include_background=True,ignore_empty=False)
            masd = monai.metrics.SurfaceDistanceMetric(include_background=False, symmetric = True)
            nsd = monai.metrics.SurfaceDiceMetric(include_background=False, distance_metric="euclidean", class_thresholds=[2])
            preds_mask = torch.tensor(preds_3d[uid]).unsqueeze(0).unsqueeze(0)
            true_mask = torch.tensor(masks_3d[uid]).unsqueeze(0).unsqueeze(0)
            print(uid)
            dice_val = dice(preds_mask,true_mask).item()
            masd_val = masd(preds_mask,true_mask).item()
            nsd_val = nsd(preds_mask,true_mask).item()
            dice_l.append(dice_val)
            masd_l.append(masd_val)
            nsd_l.append(nsd_val)
            print("Dice:",dice_val)
            print("MASD:",masd_val)
            print("NSD:",nsd_val)
print()
print("Average Metrics")
print("Dice",np.mean(dice_l))
print("MASD",np.mean(masd_l))
print("NSD",np.mean(nsd_l))

##############


# (actual height - target height) // 2
# 

# print("---")

# print(np.array(preds_3d["436"]).shape)
# print(np.array(masks_3d["436"]).shape)

# for uid in uids:
#     print("UID:",uid)
#     dice = monai.metrics.DiceMetric(include_background=True,ignore_empty=False)
#     masd = monai.metrics.SurfaceDistanceMetric(include_background=False, symmetric = True)
#     nsd = monai.metrics.SurfaceDiceMetric(include_background=False, distance_metric="euclidean", class_thresholds=[2])
#     preds_mask = torch.tensor(preds_3d[uid]).unsqueeze(0).unsqueeze(0)
#     true_mask = torch.tensor(masks_3d[uid]).unsqueeze(0).unsqueeze(0)
#     print(dice(preds_mask,true_mask))
#     print(masd(preds_mask,true_mask))
#     print(nsd(preds_mask,true_mask))
# print(dice(np.expand_dims(preds_3d["436"],axis=0),np.expand_dims(masks_3d["436"],axis=0)))

# print(np.array(preds_3d["436"]).max())
# print(np.array(preds_3d["436"]).min())
# a1 = torch.rand(1,256,256)
# a2 = torch.rand(1,256,256)

# k = []
# k.append(a1)
# k.append(a2)
# print(np.array(k).shape)

# scp 10.1.45.57:/home/lakes/bonbid-2024/checkpoints/2D_se_resnext101_32x4d/models/model_epoch_67_se_resnext101_32x4d.pth /Users/amograo/Research_Projects/DL_HIE_2024/