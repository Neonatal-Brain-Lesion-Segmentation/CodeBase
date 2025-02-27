# %%
import torch
torch.cuda.empty_cache()
import torch.nn as nn
from torch.utils.data import DataLoader
# from torchvision import transforms

import segmentation_models_pytorch as smp
import monai

from data_organization import HIE_Dataset
from pipeline_utils import *
from transforms.preprocess_v3 import transform_2d, augment

from models.vision_transformer import SwinUnet
from models.config import get_config

import os
import wandb

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchio")

# %%
WANDB_ON = False
RESUME = 0
RESUME_EPOCH = None
NUM_EPOCHS = 10
num_channels = 2
mode = '2D'
# ENCODER = "se_resnext101_32x4d"
# ENCODER = "efficientnet-b5"
# ENCODER = "densenet161"
# ENCODER = "inceptionv4"
ENCODER = "swin"
BATCH_SIZE = 16


DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
print(DEVICE)

DEST_DIR = f"./{mode}_{ENCODER}"
make_checkpoint_dir(DEST_DIR)   

DATA_ROOT = "../DATASET"
print(os.getcwd())
print(os.listdir(DATA_ROOT))

if WANDB_ON:
    wandb.init(project=f"{mode.upper()}-Segmentation-{ENCODER}-DiceFocal-Stacked",
            name = "Run-1",
                config={
            "mode": mode,
            "learning_rate": 0.0001,
            "architecture": ENCODER,
            "dataset": "BONBID-2024",
            "epochs": NUM_EPOCHS,
            "Message":"Changing LRs"
        },)

# %%

config_path = "codebase/models/config.yaml"  # Replace with the path to the new YAML file
config = get_config(config_path)

model = SwinUnet(
    config=config,          # Model configuration
    img_size=config.DATA.IMG_SIZE,
    num_classes=1,          # Number of output classes
    zero_head=False,        # Whether to use a zero-initialized head
    vis=False               # Enable visualization mode if needed
)

model.to(DEVICE)

# loss = monai.losses.DiceLoss(sigmoid=False)
loss = monai.losses.DiceFocalLoss()

metrics = [
            ('Dice', monai.metrics.DiceMetric(include_background=True,ignore_empty=False)),  
            ('IoU', monai.metrics.MeanIoU(include_background=True,ignore_empty=False)), 
        ]

metrics_3d = [
    ('Dice',monai.metrics.DiceMetric(include_background=True,ignore_empty=False),0.0),
    ('MASD',monai.metrics.SurfaceDistanceMetric(include_background=False, symmetric = True),np.inf),
    ('NSD',monai.metrics.SurfaceDiceMetric(include_background=False, distance_metric="euclidean", class_thresholds=[2]),0.0)
]
metrics_best_3d_dict = {i[0]:i[2] for i in metrics_3d}

optimizer = torch.optim.Adam(
    [
        dict(params=model.parameters(), lr=0.0001),
    ]
)

best_score = 0
best_loss = np.inf
START_EPOCH = -1

if RESUME:
    START_EPOCH, best_score, best_loss, metrics_best_3d_dict = resume_checkpoint(f"{DEST_DIR}/models",model,optimizer,DEVICE,epoch=RESUME_EPOCH,string=f"_{ENCODER}",return_list=["Epoch","Best Score","Best Loss"],return_dict=["Best 3D Dice","Best 3D MASD","Best 3D NSD"])


# optimizer.param_groups[0]["lr"] = 0.5*0.0001
print("LR:",optimizer.param_groups[0]["lr"])
# %%
train_dataset = HIE_Dataset(
    images_dir = [f'{DATA_ROOT}/BONBID2024_Train/ADC',f'{DATA_ROOT}/BONBID2024_Train/Z_ADC'] if num_channels == 2 else [f'{DATA_ROOT}/BONBID2024_Train/Z_ADC'],
    masks_dir = f'{DATA_ROOT}/BONBID2024_Train/LABEL',
    csv_file = f'{DATA_ROOT}/BONBID2024_Train/metadata.csv',
    dimension = mode,
    transform=transform_2d,
    augment=augment
)

val_dataset = HIE_Dataset(
    images_dir = [f'{DATA_ROOT}/BONBID2024_Val/ADC',f'{DATA_ROOT}/BONBID2024_Val/Z_ADC'] if num_channels == 2 else [f'{DATA_ROOT}/BONBID2024_Val/Z_ADC'],
    masks_dir = f'{DATA_ROOT}/BONBID2024_Val/LABEL',
    csv_file = f'{DATA_ROOT}/BONBID2024_Val/metadata.csv',
    dimension = mode,
    transform=transform_2d
)

img = train_dataset[33][0]
print(img.shape,img[0].min(),img[0].max(),'\n',img[1].min(),img[1].max())

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# %%

header = ["Epoch"]+["Train "+i[0] for i in metrics]+["Train Loss"]+["Val "+i[0] for i in metrics]+["Val Loss"]+["Best Score", "Best Loss","LR"]+["Val 3D "+i[0] for i in metrics_3d]+["Best 3D "+i[0] for i in metrics_3d]
stats_df = pd.DataFrame(columns=header)

for epoch in range(START_EPOCH+1,START_EPOCH+NUM_EPOCHS+1):
    print("Epoch:", epoch)
    train_logs = epoch_runner("train", train_loader, model, loss, metrics, optimizer, DEVICE)
    valid_logs = epoch_runner("val", val_loader, model, loss, metrics, device=DEVICE)
    inference_3d_logs = inference_3d_runner(val_dataset.images_dir,val_dataset.masks_dir,val_dataset.ids,val_dataset.mode,model,metrics_3d,DEVICE)
    
    best=False
    best_3d = False
    if valid_logs[metrics[0][0]] >= best_score:
        best_score = valid_logs[metrics[0][0]]
        best = True
    
    if valid_logs['Loss'] <= best_loss:
        best_loss = valid_logs['Loss']
        best = True
    
    for metric in metrics_3d:
        if metric[2] == 0:
            if inference_3d_logs[metric[0]] >= metrics_best_3d_dict[metric[0]]:
                metrics_best_3d_dict[metric[0]] = inference_3d_logs[metric[0]]
                best_3d = '_3d'
        else:
            if inference_3d_logs[metric[0]] <= metrics_best_3d_dict[metric[0]]:
                metrics_best_3d_dict[metric[0]] = inference_3d_logs[metric[0]]
                best_3d = '_3d'

    checkpoint = append_metrics_to_df(stats_df, (train_logs, "Train "), (valid_logs, "Val "), (inference_3d_logs,"Val 3D "),
                                      ({"Epoch": epoch, "Best Score": best_score, "Best Loss": best_loss, "LR":optimizer.param_groups[0]["lr"]}, ""), 
                                      (metrics_best_3d_dict,"Best 3D "))
    stats_df.to_csv(f"{DEST_DIR}/logs/stats_{START_EPOCH+1}_{ENCODER}.csv", index=False)
    
    if WANDB_ON:
        wandb.log(checkpoint)

    checkpoint["model_state_dict"] = model.state_dict()
    checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    checkpoint["batch_size"] = train_loader.batch_size
    checkpoint["encoder"] = ENCODER

    torch.save(checkpoint, f"{DEST_DIR}/models/latest_model_{ENCODER}.pth")
    if best or best_3d:
        print(f"A best model{best_3d if best_3d else ''} has been saved!")
        torch.save(checkpoint, f"{DEST_DIR}/models/model_epoch_{epoch}_{ENCODER}{best_3d if best_3d else ''}.pth")

if WANDB_ON:
    wandb.finish()




