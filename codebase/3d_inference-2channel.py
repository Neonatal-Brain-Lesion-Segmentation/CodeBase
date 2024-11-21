import torch
import segmentation_models_pytorch as smp
import monai

from data_organization import HIE_Dataset, reassemble_to_3d
from pipeline_utils import *
from transforms.preprocess_v3 import transform_2d_inner, padding

import pandas as pd
import numpy as np


DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
print(DEVICE)
# ENCODER = "se_resnext101_32x4d"
ENCODER = "densenet161"
DATA_ROOT = "/Users/amograo/Desktop/DATASET"

model = smp.Unet(
    encoder_name=ENCODER,
    encoder_weights=None,     
    in_channels=2,             
    classes=1,
    activation='sigmoid'               
)

model.to(DEVICE)

checkpoint = torch.load("/Users/amograo/Downloads/model_epoch_180_densenet161_3d.pth",map_location=torch.device(DEVICE))    
model.load_state_dict(checkpoint['model_state_dict'])


df = pd.read_csv(f'{DATA_ROOT}/BONBID2024_Val/metadata.csv')
uids = [str(i).zfill(3) for i in df["Patient ID"].tolist()]
print(uids)

preds_3d = {i: [] for i in uids}
masks_3d = {uid: reassemble_to_3d(f'{DATA_ROOT}/BONBID2024_Val/LABEL', uid) for uid in uids}


#######

dice_l = []
masd_l = []
nsd_l = []
model.eval()
image_paths = [f'{DATA_ROOT}/BONBID2024_Val/ADC',f'{DATA_ROOT}/BONBID2024_Val/Z_ADC'] #f'{DATA_ROOT}/BONBID2024_Val/ADC',
for uid in uids:
    image_set = [reassemble_to_3d(path, uid) for path in image_paths]
    # image_set = reassemble_to_3d(f'{DATA_ROOT}/BONBID2024_Val/Z_ADC', uid)

    for i in range(image_set[0].shape[0]):
        image = np.expand_dims(transform_2d_inner(np.stack([image_set[j][i] for j in range(len(image_set))]),['ADC','Z_ADC']),axis=0)
        # image = np.expand_dims(resample(np.stack([image_set[i]])),axis=0)

        image = torch.tensor(image).to(DEVICE)

        output = model(image)
        pred = (output >= 0.5).float()

        # print(image_set[0].shape)
        # print(list(image_set[0].shape))
        shape = image_set[0].shape
        shape = (1,shape[1],shape[2])
        # print(shape)

        preds_3d[uid].append(padding(pred.detach().cpu().numpy()[0],target_size=tuple(shape))[0])    


        # print(pred.shape)
        # print(np.array(preds_3d[uid]).shape)
        if len(preds_3d[uid]) == image_set[0].shape[0]:
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
