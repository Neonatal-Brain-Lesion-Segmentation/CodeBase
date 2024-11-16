import numpy as np
import torch
from torchvision import transforms

'''
2D and 3D Compatible Preprocessing Functions
Assumption:
Input File Format: ADC, ZADC and Label Files (.npy)
Output File Format: ADC, ZADC and Label Files (.npy)

1. ADC Noise Clipping [0 - 3400]
2. Resampling (256 x 256 x D) - Zero Padding
3. Min-Max Normalization
4. Augmentation (Rotation, Affine, Intensity, Horizontal Flip, Gaussian Noise and Blur)

! Lesion Volume Calculation
'''

def clip(data:np.ndarray, mode:str='ADC', min_clip:float = 0, max_clip:float = 3400) -> np.ndarray:
    """
    Clips the ADC values to a certain range [0, 3400] by default. Returns the data if as it is if the mode is not 'ADC'.
    """

    if mode.upper() == 'ADC':
        data = np.clip(data, min_clip, max_clip)

    return data

def resample(data:np.ndarray, padding_mode:str = 'constant', padding_value:float = 0., target_size:int = 256) -> np.ndarray:
    """
    Resizes 2D Slices (numpy arrays) to a target size (256 x 256) by zero padding
    TODO: Why not albumentations?
    """
    D, H, W = data.shape

    pad_height = max(0, target_size - H)
    pad_width = max(0, target_size - W)

    resampled_data = np.pad(data, ((0, 0), (pad_height // 2, pad_height - pad_height // 2), (pad_width // 2, pad_width - pad_width // 2)), mode = padding_mode, constant_values = padding_value)
    return resampled_data

def normalize(mode:str, data:np.ndarray) -> np.ndarray:
    """
    Min-Max Normalization of the ADC values of a 2D Slice
    """
    if mode.upper() == 'ADC':
        min_value = np.min(data)
        max_value = np.max(data)

    data = (data - min_value) / (max_value - min_value)
    return data

def augmentation(data:np.ndarray, augmentation_index:int) -> np.ndarray:
    '''
    Augmentation (ADC, ZADC and corresponding Label Image) - Rotation, Affine, Horizontal Flip, Gaussian Noise and Blur
    '''
    augmentations = {
        0: [], # No Augmentation
        1: transforms.RandomRotation(degrees = 20, fill = 0.), # "Rotation (20°)
        2: transforms.RandomAffine(degrees = 0, translate = (0.1, 0.1), fill = 0.), # Affine Transformation
        # 3: transforms.ColorJitter(brightness = 0.1, contrast = 0.1), # Intensity Transformation
        3: transforms.RandomHorizontalFlip(p = 1.0), # Horizontal Flip
        4: lambda x: x + 0.05 * torch.randn_like(x), # Gaussian Noise
        5: transforms.GaussianBlur(kernel_size=(5, 5), sigma = (0.1, 2.0)) # Gaussian Blur
    }
    if augmentation_index == 0:
        return data
    else:
        return augmentations[augmentation_index](data)
    
def brain_lesion_percentage(adc_image:np.ndarray, label_image:np.ndarray) -> float:
    """
    Calculate the percentage of lesion volume in the brain volume, from the 3D ADC and Label Image
    """
    brain_mask = np.where((adc_image >= 1) & (adc_image <= 3400), 1, 0)
    lesion_mask = np.where(label_image == 1, 1, 0)
    return np.sum(lesion_mask) / np.sum(brain_mask) * 100

'''
Per Patient and Per Slice Lesion Volume Calculation
1. per_slice_lesion_volume(

From the ADC Map of each Slice per Patient, we calculate the Total Area of Brain Tissue by Thresholding the ADC Map
From the Label Map of each Slice per Patient, we calculate the Total Area of Lesion Tissue (1 - Lesion and 0 - Non-Lesion), which can be from the Ground Truth or the Predicted Lesion Mask
Now, we can calculate the Lesion Volume Percentage per Slice by dividing the Total Area of Lesion Tissue by the Total Area of Brain Tissue

2. per_patient_lesion_volume() Function
For the per Patient Lesion Volume Calculation, we sum the Lesion Volume Percentage per Slice and divide it by the Total Area of Brain Tissue across all Slices

Input: ADC Map, Label Map (.npy file)
Output: Lesion Volume Percentage per Slice and per Patient

! Assumption: Total Brain Tissue Area is calculated from ADC Maps and .npy files is the file format

def per_patient_lesion_volume(image_adc, image_label):
    brain_volume_per_slice = []
    lesion_volume_per_slice = []
    lesion_volume_percentage_per_slice = []

    for i in range(image_adc.shape[0]):
        lesion_volume_percentage, brain_volume, lesion_volume = per_slice_lesion_volume(image_adc[i, :, :], image_label[i, :, :])
        brain_volume_per_slice.append(brain_volume)
        lesion_volume_per_slice.append(lesion_volume)
        lesion_volume_percentage_per_slice.append(lesion_volume_percentage)

    # Per Patient Lesion Volume Calculation
    brain_volume_per_patient = np.sum(brain_volume_per_slice)
    lesion_volume_per_patient = np.sum(lesion_volume_per_slice)
    lesion_volume_percentage_per_patient = (lesion_volume_per_patient / brain_volume_per_patient) * 100

    return lesion_volume_percentage_per_patient

def per_slice_lesion_volume(image_adc_slice, label_slice):
    brain_mask = make_brain_mask(image_adc_slice, lower_bound=1, upper_bound=3400)

    # Brain Volume Calculation
    brain_volume = np.sum(brain_mask)
    # Lesion Volume Calculation
    lesion_volume = np.sum(label_slice)
    # Per Slice Lesion Volume Percentage
    lesion_volume_percentage = (lesion_volume / brain_volume) * 100

    return lesion_volume_percentage, brain_volume, lesion_volume

def make_brain_mask(image_adc_slice, lower_bound = 1, upper_bound = 3400):
    brain_mask = np.where((image_adc_slice >= lower_bound) & (image_adc_slice <= upper_bound), 1, 0)
    return brain_mask
'''