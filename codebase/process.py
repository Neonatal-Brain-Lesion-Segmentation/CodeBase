"""
The following is a simple example algorithm.

It is meant to run within a container.

To run it locally, you can call the following bash script:

  ./test.sh

This will start the inference and reads from ./test/input and outputs to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./export.sh

Any container that shows the same behavior will do, this is purely an example of how one COULD do it.

Happy programming!
"""
from pathlib import Path
import json
from glob import glob
import SimpleITK
import numpy as np
import torch
import os
import torch.nn as nn
from data_organization import *
from transforms.preprocess_v3 import transform_2d_inner, padding

import segmentation_models_pytorch as smp

# INPUT_PATH = Path("/input")
# OUTPUT_PATH = Path("/output")

ENCODER = "densenet161"

model_path = "/Users/amograo/Downloads/model_epoch_180_densenet161_3d.pth"

# input_zadc_dir = "/input/images/z-score-adc"
# input_adc_ss_dir = "/input/images/skull-stripped-adc-brain-mri"
# output_path = "/output/images/hie-lesion-segmentation"
# os.makedirs(output_path, exist_ok=True)

input_zadc_dir = "/Users/amograo/Research_Projects/DL_HIE_2024/data_dir/BONBID2023_Val/2Z_ADC"
input_adc_ss_dir = "/Users/amograo/Research_Projects/DL_HIE_2024/data_dir/BONBID2023_Val/1ADC_ss"
output_path = "/Users/amograo/Research_Projects/DL_HIE_2024/output/images/hie-lesion-segmentation"

os.makedirs(output_path, exist_ok=True)
# output

## SET A DEVICE
def get_default_device():
    ######## set device#########
    if torch.cuda.is_available():
        print ("Using gpu device")
        return torch.device('cuda')
    else:
        print ("Using cpu device")
        return torch.device('cpu')

def run():
  DEVICE = get_default_device()

  ## ADC AND ZADC FILES

  zadc_file_paths: list[Path] = list(Path(input_zadc_dir).glob('*.mha'))
  adc_file_paths: list[Path] = list(Path(input_adc_ss_dir).glob('*.mha'))

  ## SORTING FOR CORRESPONDING MATCHING
  zadc_file_paths.sort()
  adc_file_paths.sort()


  for i in zip(adc_file_paths, zadc_file_paths):
      print(i[0].name.replace('-ADC_ss','_lesion'))
      # print(str(i[0]).split('/')[-1])
      break
      print()
  ####
  # MODEL DEFINITION AND LOADING STATE DICT
  model = smp.Unet(
      encoder_name = ENCODER,
      encoder_weights = None,
      in_channels = 2,
      classes = 1,
      activation = 'sigmoid')

  model.to(DEVICE)

  checkpoint = torch.load(model_path,map_location=DEVICE)
  model.load_state_dict(checkpoint['model_state_dict'])

  ## START MODEL EVAL
  model.eval()
  for adc_ss, zadc in zip(adc_file_paths, zadc_file_paths):
      ## GET IMAGE ARRAYS AND SPACING[::-1]
      zadc_image_array, zadc_spacing, _, _ = extract_mha_file(zadc)
      adc_image_array, adc_spacing, _, _ = extract_mha_file(adc_ss)

      ## SAVE FILE NAME
      save_name = adc_ss.name.replace('-ADC_ss','_lesion')

      ## NOTING ORIGINAL SHAPE
      image_shape = zadc_image_array.shape

      ## 3D IMAGE TO BE CREATED
      predicted_3d = []

      ## ITERATING OVER EACH SLICE
      for i in range(image_shape[0]):
          ## STACKING ADC AND ZADC
          stacked = np.stack([adc_image_array[i], zadc_image_array[i]], axis=0)
          ## TRANSFORMATION -> PADDING, CLIPING, NORMALIZING
          transformed = transform_2d_inner(stacked,['ADC','Z_ADC'])
          ## ADDING BATCH SIZE AS 1 -> [2,256,256] -> [1,2,256,256]
          transformed = transformed.unsqueeze(0)
          ## PASS IT TO DEVICE
          transformed = transformed.to(DEVICE)

          ## MODEL PREDICTION
          output = model(transformed)
          ## THRESHOLDING
          predicted_mask = (output >= 0.5).float()
          ## DETACHING AND SQUEEZING (REMOVING BATCH DIMENSION)
          predicted_mask = predicted_mask.detach().cpu().numpy().squeeze(0)
          ## REMOVING THE PADDING
          resized_predicted_mask = padding(predicted_mask, (1, image_shape[1], image_shape[2])).squeeze()
          ## APPENDING TO PREDICTED 3D
          predicted_3d.append(resized_predicted_mask)
      
      predicted_3d = np.array(predicted_3d)

      save_img_object = sitk.GetImageFromArray(predicted_3d)
      sitk.WriteImage(save_img_object, f"{output_path}/{save_name}")
      # save the numpy array as mha





if __name__ == "__main__":
    raise SystemExit(run())