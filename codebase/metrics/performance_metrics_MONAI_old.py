import torch
import numpy as np
import monai
from monai.metrics import DiceMetric, compute_average_surface_distance, compute_surface_dice
from monai.transforms import EnsureType
import SimpleITK as sitk

def calculate_metrics_monai(predicted, ground_truth, dilation=3):
    # Convert images to torch tensors and ensure they are binary
    predicted = torch.tensor(predicted, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    ground_truth = torch.tensor(ground_truth, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    if torch.sum(predicted) == 0 and torch.sum(ground_truth) == 0:
        return 0, 1, 1   # masd, nsd, dice
    elif torch.sum(predicted) == 0 or torch.sum(ground_truth) == 0:
        return 0, 1, 0   # masd, nsd, dice

    # Ensure the type is correct for MONAI metrics
    predicted = EnsureType()(predicted)
    ground_truth = EnsureType()(ground_truth)

    # Compute Dice Score
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    dice_metric(y_pred=predicted, y=ground_truth)
    # msd = monai.metrics.SurfaceDistanceMetric(include_background=False, distance_metric="euclidean")
    dice = dice_metric.aggregate().item()  # Aggregate the result

    # Compute MASD using 
    masd = compute_average_surface_distance(y_pred=predicted, y=ground_truth, 
                                            distance_metric="euclidean").mean().item()

    # Compute NSD 
    nsd = compute_surface_dice(y_pred=predicted, y=ground_truth,
                                class_thresholds=[dilation], distance_metric="euclidean").mean().item()

    return masd, nsd, dice


if __name__ == "__main__":
    image_path1 = '../BONBID2023_Train/3LABEL/MGHNICU_014-VISIT_01_lesion.mha'  
    image1 = sitk.ReadImage(image_path1)
    image_array1 = sitk.GetArrayFromImage(image1) 
    
    image_path2 = '../BONBID2023_Train/3LABEL/MGHNICU_405-VISIT_01_lesion.mha' 
    image2 = sitk.ReadImage(image_path2)
    image_array2 = sitk.GetArrayFromImage(image2) 

    # Calculating performance metrics 3D
    masd, nsd, dice = calculate_metrics_monai(image_array1, image_array2)
    print("3D Metrics")
    print(f'Mean Average Surface Distance (MASD): {masd}')
    print(f'Normalized Surface Distance (NSD): {nsd}')
    print(f'Dice Coefficient: {dice}\n')

    # Calculating performance metrics 2D slices
    total_masd, total_nsd, total_dice = 0.0, 0.0, 0.0
    for i in range(image_array1.shape[0]):
        masd, nsd, dice = calculate_metrics_monai(image_array1[i], image_array2[i])
        total_masd += masd
        total_nsd += nsd
        total_dice += dice
    print("Mean of 2D Slices Metrics")
    print(f'Mean Average Surface Distance (MASD): {total_masd/image_array1.shape[0]}')
    print(f'Normalized Surface Distance (NSD): {total_nsd/image_array1.shape[0]}')
    print(f'Dice Coefficient: {total_dice/image_array1.shape[0]}')

