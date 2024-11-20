import numpy as np
from scipy.ndimage import zoom
import torchio as tio
import random

'''
Preprocessing Functions (ADC, ZADC and Label Files (.npy) 2D and 3D Compatible)
1. Clipping (ADC: [0 - 3400])
2. Resampling (256 x 256 x D) - Zero Padding] for 2D and 3D
    ! Anti-Resampling
3. Interpolation (256 x 256 x 64) - Varying Types [This should also include updating the Spacing and Origin of the Image] for 3D
   ! Anti-Interpolation
4. Central Patch Extraction (224 x 224)
4. Min-Max Normalization (explore Histogram Normalization)
5. Augmentation
'''

def clip(mode:str, data:np.ndarray, min_clip:int = 0, max_clip:int = 3400) -> np.ndarray:
    """
    Clips ADC Maps between a range of [min_clip, max_clip]
    Args:
        mode (str): If 'ADC', the function clips the input data otherwise, the data is returned unchanged.
        data (np.ndarray): Input Array
        min_clip (int, optional): Minimum value for clipping (Default - 0)
        max_clip (int, optional): Maximum value for clipping (Default - 3400)
    
    Returns:
        np.ndarray: Clipped Array
    """
    if mode.upper() == 'ADC':
        return np.clip(data, min_clip, max_clip)
    else:
        return data
    
def padding(data:np.ndarray,
            target_size:tuple = (1, 256, 256), 
            padding_mode:str = 'constant', 
            padding_value:float = 0.) -> np.ndarray:
    """
    Resizes (Up and Downsampling) 2D Slices from Input Data (1 x H x W) to a target size (1 x H x W) by zero padding
    Args:
        data (np.ndarray): Input Array (1 x H x W)
        target_size (tuple, optional): Target Size (1 x H x W) (Default: (1, 256, 256))
        padding_mode (str, optional): Padding Mode (Default: 'constant')
        padding_value (float, optional): Padding Value (Default: 0.)

    Returns:
        np.ndarray: Padded or Cropped Array (1 x H x W) 
    """
    D, H, W = data.shape
    target_depth, target_height, target_width = target_size

    # if D != target_depth:
    #     raise ValueError(f"Invalid Depth: {D} for Target Depth: {target_depth}")
    
    # Padding and Cropping Height and Width
    pad_h, crop_h = max(0, target_height - H), max(0, H - target_height)
    pad_w, crop_w = max(0, target_width - W), max(0, W - target_width)

    # Padding
    if pad_h or pad_w:
        data = np.pad(data, ((0, 0), (pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2)), mode=padding_mode, constant_values=padding_value)

    # Cropping
    if crop_h or crop_w:
        data = data[:, crop_h // 2: H - crop_h // 2, crop_w // 2: W - crop_w // 2]

    return data

'''
Resampling (D x H x W) 
1. Nearest-Neighbor Interpolation (Order = 0)
2. Linear Interpolation (Order = 1)
3. Spline Interpolation (Order = 7) 
'''
def resample(data:np.ndarray, # Input Data (D x H x W), where D = 1 (2D) or D > 1 (3D)
             target_shape:tuple[int, ...], # If 2D (H x W), If 3D (D x H x W) Accepted
             input_spacing:tuple[float, float, float], # Spacing [Axial, Coronal, Sagittal]
             mode: str = '2D' or '3D', # Mode (2D or 3D)
             interpolate_order: int = 0, # Interpolation Order (0: Nearest Neighbor, 1: Trilinear, 2: Tri-Bicubic, 3: Tri-Quartic)
             interpolation_mode: str = 'constant', # Interpolation Mode - To handle 
             interpolation_cval: float = 0.0) -> tuple[np.ndarray, tuple[float, float, float]]:
    """
    Resampling (Up and Downsampling) 2D Slices and 3D Volumes to a Target Shape (D x H x W) by Interpolation Methods (Nearest Neighbor, Linear, Spline)
    Args:
        mode (str): Input Data Type - '2D' or '3D'
        data (np.ndarray): Input Array
            - For 2D: shape (1 x H x W)
            - For 3D: shape (D x H x W)
        target_shape (tuple[int, ...]): Target Dimensions
            - For 2D: (H x W)
            - For 3D: (D x H x W)
        input_spacing (tuple[float, float, float], optional): Voxel Spacing [Axial, Coronal, Sagittal])
        interpolate_order (int, optional): (Default: 0)
            - 0: Nearest Neighbor interpolation
            - 1: Trilinear interpolation
            - 2: Tri-Bicubic interpolation
            - 3: Tri-Quartic interpolation
        interpolation_mode (str, optional): How to handle values outside the grid? (Default: 'constant')
            Options: 'constant', 'nearest', 'reflect', 'mirror', 'wrap'
        interpolation_cval (float, optional): Value to be filled past edges if mode is 'constant' (Default: 0.0)

    Returns:
        tuple[np.ndarray, tuple[float, float, float]]: Tuple(Resampled Image, Updated Spacing)
            - resampled_image (np.ndarray): Resampled Array with Output Dimensions
                - For 2D: shape (1 x target_H x target_W)
                - For 3D: shape (target_D x target_H x target_W)
            - output_spacing (tuple[float, float, float]): Updated Spacing after Resampling [Axial, Coronal, Sagittal]
    """
    # Shape of the Input Data (2D [1 x H x W] or 3D [D x H x W])
    D, H, W = data.shape

    # If 2D, Add a Channel Dimension (1 x H x W)
    if mode.upper() == '2D' and D == 1:
        target_shape = (D, *target_shape)

    # Scaling Factor for Resampling
    scale_factor = tuple(t / o for t, o in zip(target_shape, data.shape))
    # Interpolate
    resampled_image = zoom(input = data, zoom = scale_factor, order = interpolate_order, mode = interpolation_mode, cval = interpolation_cval)
    # Update Spacing
    output_spacing = tuple(os / sf for os, sf in zip(input_spacing, scale_factor))

    return (resampled_image, output_spacing)

def min_max_normalize(data:np.ndarray, mode:str) -> np.ndarray:
    """
    Min-Max Normalization of 2D (1 x H x W) and 3D (D x H x W) ADC Maps
    """
    if mode.upper() == 'ADC':
        min_value = np.min(data)
        max_value = np.max(data)
        
        # data = (data - min_value) / (max_value - min_value)
        return (data - min_value) / (max_value - min_value + 1e-9)
    return data

def transform_2d(input_array:np.ndarray,mode_list:list[str]):

    if input_array.shape[0] != len(mode_list):
        raise ValueError(f"Invalid Input Array Shape: {input_array.shape} for Mode List: {mode_list}")
    
    padded = padding(input_array)
    new = []
    for idx,mode in enumerate(mode_list):
        new.append(min_max_normalize(clip(mode,padded[idx]),mode))

    return np.array(new)

def augmentations(data: np.ndarray) -> np.ndarray:
    """
        Random Augmentations (Flip, Anisotropy, Blur, Noise, Gamma)
        Args:
            data (np.ndarray): 2d or 3D Image Array
        Returns:
            np.ndarray: Augmented Image Array
    """
    transformations = {
        0: tio.Compose([]),  # No augmentation
        1: tio.Compose([tio.transforms.RandomFlip(axes=(0), flip_probability=1.0)]),  # Inverse the Stack
        2: tio.Compose([tio.transforms.RandomFlip(axes=(1), flip_probability=1.0)]),  # Vertical Flip
        3: tio.Compose([tio.transforms.RandomFlip(axes=(2), flip_probability=1.0)]),  # Horizontal Flip
        4: tio.Compose([tio.transforms.RandomFlip(axes=(0, 1), flip_probability=1.0)]),  # Vertical Flip and Inversed Stack
        5: tio.Compose([tio.transforms.RandomFlip(axes=(0, 2), flip_probability=1.0)]),  # Horizontal Flip and Inversed Stack
        6: tio.Compose([tio.transforms.RandomFlip(axes=(1, 2), flip_probability=1.0)]),  # Horizontal Flip and Vertical Flip
        7: tio.Compose([tio.transforms.RandomFlip(axes=(0, 1, 2), flip_probability=1.0)]),  # Horizontal Flip, Vertical Flip, and Inversed Stack
        8: tio.Compose([tio.transforms.RandomAnisotropy(downsampling=(1.1, 1.5), p=1.0)]),  # Anisotropy
        9: tio.Compose([tio.transforms.RandomBlur(std=(0, 0.4), p=1.0)]),  # Blur
        10: tio.Compose([tio.transforms.RandomNoise(std=(0, 0.01), p=0.5)]),  # Noise
        11: tio.Compose([tio.transforms.RandomNoise(std=(0, 0.05), p=0.5)]),  # Noise
        12: tio.Compose([tio.transforms.RandomGamma(log_gamma=(-0.3, 0.3), p=1.0)]),  # Gamma
        13: tio.Compose([tio.transforms.RandomGamma(log_gamma=(-0.1, 0.1), p=1.0)]),  # Gamma
        14: tio.Compose([tio.transforms.RandomAnisotropy(downsampling=(1.1, 1.5), p=1.0), tio.transforms.RandomFlip(axes=(1), flip_probability=1.0)]),  # Anisotropy and Vertical Flip
        15: tio.Compose([tio.transforms.RandomBlur(std=(0, 0.4), p=1.0), tio.transforms.RandomFlip(axes=(2), flip_probability=1.0)]),  # Blur and Horizontal Flip
        16: tio.Compose([tio.transforms.RandomNoise(std=(0, 0.01), p=0.5), tio.transforms.RandomFlip(axes=(1, 2), flip_probability=1.0)]),  # Noise and Horizontal Flip, Vertical Flip, and Inversed Stack
        17: tio.Compose([tio.transforms.RandomGamma(log_gamma=(-0.3, 0.3), p=1.0), tio.transforms.RandomFlip(axes=(0, 1, 2), flip_probability=1.0)]),  # Gamma and Horizontal Flip, Vertical Flip, and Inversed Stack
        18: tio.Compose([tio.transforms.RandomGamma(log_gamma=(-0.1, 0.1), p=1.0), tio.transforms.RandomFlip(axes=(0, 1), flip_probability=1.0)])  # Gamma and Horizontal Flip, Vertical Flip, and Inversed Stack
    }

    random_key = random.randint(0, len(transformations) - 1)
    transformed_data = transformations[random_key](data)
    return transformed_data