# loop through all .mha files in directory ../BONBID2023_Train/3LABEL/ and get array with sitk

import os
import SimpleITK as sitk

directory = '../BONBID2023_Train/3LABEL/'
for filename in os.listdir(directory):
    if filename.endswith(".mha"):
        image_path = os.path.join(directory, filename)
        image = sitk.ReadImage(image_path)
        image_array = sitk.GetArrayFromImage(image)  # Convert to a numpy array (shape: [depth, height, width])
        # print name and shape
        print(f'File: {filename} ||||| Shape: {image_array.shape}')
    else:
        continue