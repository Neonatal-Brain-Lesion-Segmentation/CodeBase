import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils
from segmentation_models_pytorch.utils.base import Loss

from transforms.preprocess import resample


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from torchvision import transforms

from data_organization import HIE_Dataset


model = smp.Unet(
    encoder_name="densenet121",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights=None,     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=1,                  # model input channels (1 for grayscale images, 3 for RGB, etc.)
    classes=1,                      # model output channels (number of classes in your dataset)
)

loss = smp.utils.losses.DiceLoss()
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
    smp.utils.metrics.Fscore(threshold=0.5),
]
optimizer = torch.optim.Adam(
    [
        dict(params=model.parameters(), lr=0.0001),
    ]
)

# resize = transforms.Resize((256, 256))


train_dataset = HIE_Dataset(
    images_dir = ['/Users/amograo/Research_Projects/DL_HIE_2024/data_dir/BONBID2023_Train/2Z_ADC'],
    masks_dir = '/Users/amograo/Research_Projects/DL_HIE_2024/data_dir/BONBID2023_Train/3Label',
    csv_file = '/Users/amograo/Research_Projects/DL_HIE_2024/data_dir/BONBID2023_Train/df.csv',
    mode = '2d',
    transform=resample
)

val_dataset = HIE_Dataset(
    images_dir = ['/Users/amograo/Research_Projects/DL_HIE_2024/data_dir/BONBID2023_Val/2Z_ADC'],
    masks_dir = '/Users/amograo/Research_Projects/DL_HIE_2024/data_dir/BONBID2023_Val/3Label',
    csv_file = '/Users/amograo/Research_Projects/DL_HIE_2024/data_dir/BONBID2023_Val/df.csv',
    mode = '2d',
    transform=resample
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

train_epoch = smp.utils.train.TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device='mps',
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device='mps',
    verbose=True,
)

for i in range(5):
    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(val_loader)

    print('Training Logs:', train_logs)
    print('Validation Logs:', valid_logs)