import SimpleITK as sitk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def extract_mha_file(file_path:str,calc_volume=False):
    """This function disassembles a mha file and returns a numpy array, the spacing, the direction and the origin of the image. 
    If required, this function will be modified to return other parameters mentioned in the the metadata of the mha file."""
    image = sitk.ReadImage(file_path)
    image_array = sitk.GetArrayFromImage(image)
    spacing = image.GetSpacing()
    direction = image.GetDirection()
    origin = image.GetOrigin()
    volume = calculate_volume_percentage(image_array) if calc_volume else 0
    return image_array, spacing[::-1], volume, direction[::-1], origin[::-1]

def save_slices(dest_dir,image_id,image_array,category = '') -> None:
    """Saves 2d Slices as npy files of 3d Image"""
    for i in range(image_array.shape[0]):
        np.save(f"{dest_dir}/{image_id}_{category.upper()}_slice_{i}.npy", image_array[i])

def reassemble_to_3d(folder_path, uid) -> np.ndarray:
    """Reads the npy files of a certain patient and stacks them into a 3d image"""
    files = sorted([file for file in os.listdir(folder_path) if extract_id(file)==uid], key = lambda x:int(x.split('.')[0].split('_')[-1]))
    # files = sorted(os.listdir(folder_path), key = lambda x:int(x.split('.')[0].split('_')[-1]))
    slices = []
    for file in files:
        slices.append(np.load(f"{folder_path}/{file}"))
    return np.stack(slices)

def extract_id(file_name:str) -> str:
    """The assumption is that the patient ID is the fist numeric sequence in the file name."""
    elements = file_name.split('-')[0].split('_')
    for i in elements:
        if i.isdigit():
            return i

def calculate_volume_percentage(mask):
    return 0

def split_files_gen_csv(source_dir:str, dest_dir:str, category:str, gen_csv:bool=False):
    """Saves 3d files as 2d npy files from a given directory. Can caclulate volume if masks have been provided. 
    Will generate a CSV containing metadata."""

    meta_df=pd.DataFrame(columns=["Patient ID","Axial Slices", "Coronal Slices", "Sagittal Slices", "Lesion Percentage","Axial Spacing", "Coronal Spacing", "Sagittal Spacing"])

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    for file in os.listdir(source_dir):
        if not file.endswith('.mha'):
            continue

        image_array, spacing, volume, direction, origin = extract_mha_file(f"{source_dir}/{file}")
        uid = extract_id(file)

        save_slices(dest_dir,uid,image_array,category)

        if gen_csv:
            num_axial, num_coronal, num_sagittal = image_array.shape
            spacing_axial, spacing_coronal, spacing_sagittal = spacing
            meta_df.loc[len(meta_df.index)] = [uid, num_axial, num_coronal, num_sagittal, volume, spacing_axial, spacing_coronal, spacing_sagittal]
        
    meta_df.to_csv(f"{dest_dir}/metadata.csv", index=False)


