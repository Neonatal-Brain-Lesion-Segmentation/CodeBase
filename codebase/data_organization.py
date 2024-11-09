import SimpleITK as sitk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
# import dataset from torch
import torch
from torch.utils.data import Dataset

def extract_mha_file(file_path:str,calc_volume=False) -> tuple:
    """This function disassembles an MHA file and returns a numpy array, the spacing, the direction and the origin of the image. 
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

def split_files_gen_csv(source_dir:str, dest_dir:str, category:str, gen_csv:bool=False)->None:
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

class Dataset_Sample(BaseDataset):

    def __init__(
        self,
        root,
        images_dir,
        masks_dir,
        csv,
        aug_fn=None,
        id_col="DICOM",
        aug_col="Augmentation",
        preprocessing_fn=None,
    ):
        images_dir = os.path.join(root, images_dir)
        masks_dir = os.path.join(root, masks_dir)
        df = pd.read_csv(os.path.join(root, csv))
        #df = df[df["Pneumothorax"] == 1]

        self.ids = [(r[id_col], r[aug_col]) for i, r in df.iterrows()]
        self.images = [os.path.join(images_dir, item[0] + ".png") for item in self.ids]
        self.masks = [
            os.path.join(masks_dir, item[0] + "_mask.png") for item in self.ids
        ]
        self.aug_fn = aug_fn
        self.preprocessing_fn = preprocessing_fn

    def __getitem__(self, i):

        image = cv2.imread(self.images[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = (cv2.imread(self.masks[i], 0) == 255).astype("float")
        mask = np.expand_dims(mask, axis=-1)

        #image = image.astype(np.float32)

        aug = self.ids[i][1]
        # if aug:
        augmented = self.aug_fn(aug)(image=image, mask=mask)
        image, mask = augmented["image"], augmented["mask"]

        if self.preprocessing_fn:
            sample = self.preprocessing_fn(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        return image, mask

    def __len__(self):
        return len(self.ids)


class HIE_ADC_Dataset(Dataset):
    """HIE Dataset Class
    Has a 2d and a 3d option. Both are segmentation tasks. 
    In the 2d option, numpy files are given, in the 3d otpion, numpy files are constructed.
    If channels are two, then it is [ADC,ZADC]
    """
    def __init__(
            self,
            images_dir:list[tuple[str,str]],
            masks_dir:str,
            csv_file:str,
            mode:str = '2d',
            transform:function|None = None
        ):

        
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.df = pd.read_csv(csv_file)
        self.mode = mode.lower()

        self.ids = self.df['Patient ID'].values
        self.channels = len(self.images_dir)

        if self.mode == '2d':
            # self.images = [i for i in os.listdir(images_dir) if i.endswith('.npy')]
            self.images = list(zip(*[sorted([i for i in os.listdir(path) if i.endswith('.npy')]) for path in self.images_dir]))
            self.masks = sorted([i for i in os.listdir(self.masks_dir) if i.endswith('.npy')])
        else:
            self.images = self.ids
            self.masks = self.ids
        
        def __getitem__(self, i):
            if mode == '2d':
                image = np.stack([np.load(f"{self.images_dir[n]}/{self.images[i][n]}") for n in range(self.channels)])
                if self.channels == 1:
                    image = np.expand_dims(image, axis=0)

                mask = np.load(f"{self.masks_dir}/{self.masks[i]}")
                mask = np.expand_dims(mask, axis=0)
                
            else:
                # image = reassemble_to_3d(images_dir, self.images[i])
                image = np.stack([reassemble_to_3d(self.images_dir[n],self.images[i]) for n in range(self.channels)])
                mask = reassemble_to_3d(self.masks_dir, self.masks[i])

            if self.transform:
                image = self.transform(image)
                mask = self.transform(mask)
            return image, mask

        def __len__(self):
            if mode == '2d':
                return len(self.images)
            else:
                return len(self.ids)
            
            
# l = []
# for pid in self.ids:
#     # get the number of axial slices for this id from the csv
#     axial_slices = self.df[self.df['Patient ID'] == pid]["Axial Slices"]
#     for i in range(axial_slices):
#         l.append(f"{pid}_{category.upper()}_slice_{i}")
