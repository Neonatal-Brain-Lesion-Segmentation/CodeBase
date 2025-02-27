import torch
import segmentation_models_pytorch as smp
import monai

from data_organization import HIE_Dataset, reassemble_to_3d
from pipeline_utils import *
from transforms.preprocess_v3 import transform_2d_inner, padding
from transforms.preprocess_v1 import brain_lesion_percentage

from transunet.vit_seg_modeling import VisionTransformer as ViT_seg
from transunet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

def show_all_slices_in_grid(gt_masks, pred_masks, pairs_per_row=4):
    """
    Display all GT Mask and Predicted Mask slices in a grid with specified pairs per row.

    Parameters:
    gt_masks (numpy array): Ground truth masks with shape (num_slices, height, width)
    pred_masks (numpy array): Predicted masks with shape (num_slices, height, width)
    pairs_per_row (int): Number of image pairs per row
    """
    num_slices = gt_masks.shape[0]
    rows = (num_slices + pairs_per_row - 1) // pairs_per_row  # Calculate the number of rows needed
    
    fig, axes = plt.subplots(rows, pairs_per_row * 2, figsize=(20, rows * 5))
    
    for i in range(num_slices):
        row = i // pairs_per_row
        col = (i % pairs_per_row) * 2
        
        axes[row, col].imshow(gt_masks[i], cmap='gray')
        axes[row, col].set_title(f'GT Mask - Slice {i}')
        axes[row, col].axis('off')
        
        axes[row, col + 1].imshow(pred_masks[i], cmap='gray')
        axes[row, col + 1].set_title(f'Predicted Mask - Slice {i}')
        axes[row, col + 1].axis('off')
    
    # Hide any unused subplots
    for j in range(num_slices, rows * pairs_per_row):
        row = j // pairs_per_row
        col = (j % pairs_per_row) * 2
        axes[row, col].axis('off')
        axes[row, col + 1].axis('off')
    
    plt.tight_layout()
    plt.show()

def inf_3d(image_paths,uid,model,activation=False):
    dice_l = []
    masd_l = []
    nsd_l = []
    image_set = [reassemble_to_3d(path, uid) for path in image_paths]
    # image_set = reassemble_to_3d(f'{DATA_ROOT}/BONBID2024_Val/Z_ADC', uid)
    preds_3d = []
    for i in range(image_set[0].shape[0]):
        image = np.expand_dims(transform_2d_inner(np.stack([image_set[j][i] for j in range(len(image_set))]),['ADC','Z_ADC']),axis=0)
        # image = np.expand_dims(resample(np.stack([image_set[i]])),axis=0)

        image = torch.tensor(image).to(DEVICE)

        output = model(image)
        if activation==True:
            output = torch.sigmoid(output)
        # output = torch.sigmoid(output)
        pred = (output >= 0.5).float()

        # print(image_set[0].shape)
        # print(list(image_set[0].shape))
        shape = image_set[0].shape
        shape = (1,shape[1],shape[2])
        # print(shape)

        preds_3d.append(padding(pred.detach().cpu().numpy()[0],target_size=tuple(shape))[0])   

    return np.array(preds_3d).astype(np.uint8) 

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
print(DEVICE)
# ENCODER = "se_resnext101_32x4d"
ENCODER = "densenet161"
DATA_ROOT = "/Users/amograo/Desktop/DATASET"
image_paths = [f'{DATA_ROOT}/BONBID2024_Val/ADC',f'{DATA_ROOT}/BONBID2024_Val/Z_ADC'] #f'{DATA_ROOT}/BONBID2024_Val/ADC',

# model_1 = smp.UnetPlusPlus(
#     encoder_name=ENCODER,
#     encoder_weights=None,
#     in_channels=2,
#     classes=1,
#     activation='sigmoid',
# )

# model_1.to(DEVICE)

config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
config_vit.n_classes = 1
config_vit.n_skip = 3

model = ViT_seg(config_vit, img_size=256, num_classes=config_vit.n_classes)

model.to(DEVICE)

# checkpoint_1 = torch.load("/Users/amograo/Desktop/HIE-BONBID-24/UNetPP-2D_densenet161-Aug-Stacked/models/model_epoch_159_densenet161_3d.pth",map_location=torch.device(DEVICE))   
# checkpoint = torch.load("/Users/amograo/Desktop/submission/models/model_epoch_180_densenet161_3d.pth",map_location=torch.device(DEVICE))  
checkpoint = torch.load("/Users/amograo/Desktop/HIE-BONBID-24/TransUNet-3/models/model_epoch_98_TransUNet-3_3d.pth",map_location=torch.device(DEVICE))   
model.load_state_dict(checkpoint['model_state_dict'])
# model_1.load_state_dict(checkpoint_1['model_state_dict'])


df = pd.read_csv(f'{DATA_ROOT}/BONBID2024_Val/metadata.csv')
uids = [str(i).zfill(3) for i in df["Patient ID"].tolist()]
uids.sort()
print(uids)

preds_3d = {i: [] for i in uids}
masks_3d = {uid: reassemble_to_3d(f'{DATA_ROOT}/BONBID2024_Val/LABEL', uid) for uid in uids}

for uid in uids:
    preds_3d = inf_3d(image_paths,uid,model,activation=False)
    if brain_lesion_percentage()

#######

dice_l = []
masd_l = []
nsd_l = []
model.eval()


        




for uid in uids:
    image_set = [reassemble_to_3d(path, uid) for path in image_paths]
    # image_set = reassemble_to_3d(f'{DATA_ROOT}/BONBID2024_Val/Z_ADC', uid)
    preds_3d = []
    for i in range(image_set[0].shape[0]):
        image = np.expand_dims(transform_2d_inner(np.stack([image_set[j][i] for j in range(len(image_set))]),['ADC','Z_ADC']),axis=0)
        # image = np.expand_dims(resample(np.stack([image_set[i]])),axis=0)

        image = torch.tensor(image).to(DEVICE)

        output = model(image)
        output = torch.sigmoid(output)
        pred = (output >= 0.5).float()

        # print(image_set[0].shape)
        # print(list(image_set[0].shape))
        shape = image_set[0].shape
        shape = (1,shape[1],shape[2])
        # print(shape)

        preds_3d.append(padding(pred.detach().cpu().numpy()[0],target_size=tuple(shape))[0])    
    



        # print(pred.shape)
        # print(np.array(preds_3d[uid]).shape)
        if len(preds_3d[uid]) == image_set[0].shape[0]:
            preds_3d[uid] = np.stack(preds_3d[uid])
            print(brain_lesion_percentage(image_set[0],preds_3d[uid]))  
            dice = monai.metrics.DiceMetric(include_background=True,ignore_empty=False)
            masd = monai.metrics.SurfaceDistanceMetric(include_background=False, symmetric = True)
            nsd = monai.metrics.SurfaceDiceMetric(include_background=False, distance_metric="euclidean", class_thresholds=[2])
            # show_all_slices_in_grid(masks_3d[uid],preds_3d[uid])
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
