import SimpleITK as sitk
import matplotlib.pyplot as plt
import torch

image_path = "/Users/amograo/Research_Projects/DL_HIE_2024/output/images/hie-lesion-segmentation/MGHNICU_436-VISIT_01_lesion.mha"
orginal_path = "/Users/amograo/Research_Projects/DL_HIE_2024/data_dir/BONBID2023_Val/3LABEL/MGHNICU_436-VISIT_01_lesion.mha"

image = sitk.ReadImage(image_path)
array = sitk.GetArrayFromImage(image)

org_img = sitk.ReadImage(orginal_path)
org_array = sitk.GetArrayFromImage(org_img)

print(array.shape)
print(org_array.shape)

print(array.dtype)
print(org_array.dtype)
# 0.80+0.411+0.95+0.74

import monai
import numpy as np

dice_loss =monai.metrics.DiceMetric(include_background=True,ignore_empty=False)
dice = dice_loss(torch.tensor(array).unsqueeze(0).unsqueeze(0),torch.tensor(org_array).unsqueeze(0).unsqueeze(0))
print(torch.sum(dice).item())

# Plot the Original and Predicted Image (Slices side by side)
# for i in range(array.shape[0]):
#     fig, ax = plt.subplots(1, 2, figsize=(10, 5))
#     ax[0].imshow(org_array[i], cmap='gray')
#     ax[0].set_title('Original Image')
#     ax[1].imshow(array[i], cmap='gray')
#     ax[1].set_title('Predicted Image')
#     plt.show()