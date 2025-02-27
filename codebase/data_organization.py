import SimpleITK as sitk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
# import dataset from torch
import torch
from torch.utils.data import Dataset
from typing import Callable

def extract_mha_file(file_path:str) -> tuple:
    """This function disassembles an MHA file and returns a numpy array, the spacing, the direction and the origin of the image. 
    If required, this function will be modified to return other parameters mentioned in the the metadata of the mha file."""
    image = sitk.ReadImage(file_path)
    image_array = sitk.GetArrayFromImage(image)
    spacing = image.GetSpacing()
    direction = image.GetDirection()
    origin = image.GetOrigin()
    return image_array, spacing[::-1], direction[::-1], origin[::-1]

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

def calculate_volume_percentage(adc_image:np.ndarray, label_image:np.ndarray) -> float:
    """
    Calculate the percentage of lesion volume in the brain volume, from the 3D ADC and Label Image
    """
    brain_mask = np.where((adc_image >= 1) & (adc_image <= 3400), 1, 0)
    lesion_mask = np.where(label_image == 1, 1, 0)
    return np.sum(lesion_mask) / np.sum(brain_mask) * 100

def split_files_gen_csv(source_dir:str, dest_dir:str, category:str, gen_csv:bool=False, adc_dir = None)->None:
    """Saves 3d files as 2d npy files from a given directory. Can caclulate volume if masks have been provided. 
    Will generate a CSV containing metadata."""
    if gen_csv and adc_dir is not None:
        meta_df=pd.DataFrame(columns=["Patient ID","Axial Slices", "Coronal Slices", "Sagittal Slices", "Lesion Percentage","Axial Spacing", "Coronal Spacing", "Sagittal Spacing"])

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    for file in os.listdir(source_dir):
        if not file.endswith('.mha'):
            continue

        image_array, spacing, direction, origin = extract_mha_file(f"{source_dir}/{file}")
        uid = extract_id(file)

        save_slices(dest_dir,uid,image_array,category)

        if gen_csv:
            # reconstruct 3d image for adc with this UID, and pass that ND MAAK
            num_axial, num_coronal, num_sagittal = image_array.shape
            spacing_axial, spacing_coronal, spacing_sagittal = spacing
            
            adc_array = reassemble_to_3d(adc_dir, uid)
            volume = calculate_volume_percentage(adc_array, image_array)

            meta_df.loc[len(meta_df.index)] = [uid, num_axial, num_coronal, num_sagittal, volume, spacing_axial, spacing_coronal, spacing_sagittal]

    if gen_csv:
        meta_df.to_csv(f"{dest_dir}/metadata.csv", index=False)

class HIE_Dataset(Dataset):
    """HIE Dataset Class
    Has a 2d and a 3d option. Both are segmentation tasks. 
    In the 2d option, numpy files are given, in the 3d otpion, numpy files are constructed.
    If channels are two, then it is [ADC,ZADC]
    """
    def __init__(
            self,
            images_dir:list[str],
            masks_dir:str,
            csv_file:str,
            dimension:str = '2d',
            transform:callable = None,
            augment:callable = None,
            mode:list[str] = None
        ):

        # df = pd.read_csv(csv_file)
        # df = df[df['Lesion Percentage'] < 4]
        # self.df = df
        self.df = pd.read_csv(csv_file)
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.dimension = dimension.lower()
        self.mode = [i.split('/')[-1].upper() for i in images_dir] if mode == None else mode
        self.transform = transform
        self.augment = augment

        self.ids = [str(i).zfill(3) for i in self.df['Patient ID'].values]
        print(len(self.ids))
        self.channels = len(self.images_dir)

        # self.df = self.df[self.df['Lesion Percentage'] < 2.5]

        if self.dimension == '2d':
            # self.images = [i for i in os.listdir(images_dir) if i.endswith('.npy')]
            self.images = list(zip(*[sorted([i for i in os.listdir(path) if i.endswith('.npy') and i[:3] in self.ids]) for path in self.images_dir]))
            self.masks = sorted([i for i in os.listdir(self.masks_dir) if i.endswith('.npy')  and i[:3] in self.ids])
        else:
            self.images = self.ids
            self.masks = self.ids
        
    def __getitem__(self, i):

        if self.dimension == '2d':
            # C, H, W
            # print(self.images[i][2])
            image = np.stack([np.load(f"{self.images_dir[n]}/{self.images[i][n]}") for n in range(self.channels)])
            mask = np.load(f"{self.masks_dir}/{self.masks[i]}")
            mask = np.expand_dims(mask, axis=0) 
            
        else:
            # C, D, H, W
            image = np.stack([reassemble_to_3d(self.images_dir[n],self.images[i]) for n in range(self.channels)])
            mask = reassemble_to_3d(self.masks_dir, self.masks[i])
            mask = np.expand_dims(mask, axis=0)

        if self.transform:
            image, mask = self.transform(image, mask, self.mode)
        
        if self.augment:
            image, mask = self.augment(image, mask)

        return image, mask

    def __len__(self):
        return len(self.images)
    

