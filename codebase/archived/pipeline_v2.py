# %%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from torchvision import transforms

import segmentation_models_pytorch as smp
import monai

from data_organization import HIE_Dataset
from pipeline_utils import *
from transforms.preprocess import resample

import os
import wandb

# TODO: Make it such that the user can choose how they want the stacking -> ADC, ZADC or Both and in which order?

# %%

RESUME = 0
NUM_EPOCHS = 5
mode = '2D'
ENCODER = "densenet121"


DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
print(DEVICE)

DEST_DIR = f"./{mode}_{ENCODER}"
make_checkpoint_dir(DEST_DIR)   

DATA_ROOT = ""
print(os.getcwd())
print(os.listdir(DATA_ROOT))

wandb.init(project=f"{mode.upper()}-Segmentation-{ENCODER}",
           name = "Run-1",
               config={
        "mode": mode,
        "learning_rate": 0.0001,
        "architecture": ENCODER,
        "dataset": "BONBID-2024",
        "epochs": NUM_EPOCHS,
    },)

# %%

model = smp.Unet(
    encoder_name=ENCODER,
    encoder_weights=None,     
    in_channels=1,             
    classes=1,
    activation='sigmoid'               
)

model.to(DEVICE)

loss = monai.losses.DiceLoss(sigmoid=False)

metrics = [
            ('Dice', monai.metrics.DiceMetric(include_background=True,ignore_empty=False)),  
            ('IoU', monai.metrics.MeanIoU(include_background=True,ignore_empty=False)), 
        ]

optimizer = torch.optim.Adam(
    [
        dict(params=model.parameters(), lr=0.0001),
    ]
)

best_score = 0
best_loss = np.inf
START_EPOCH = -1

if RESUME:
    checkpoint = torch.load(f"{DEST_DIR}/models/latest_model_{ENCODER}.pth",map_location=torch.device(DEVICE))
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    # scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    START_EPOCH = checkpoint["Epoch"]

    best_score = checkpoint["Best Score"]
    best_loss = checkpoint["Best Loss"]


# %%
train_dataset = HIE_Dataset(
    images_dir = [f'{DATA_ROOT}/BONBID2023_Train/2Z_ADC'],
    masks_dir = f'{DATA_ROOT}/BONBID2023_Train/3Label',
    csv_file = f'{DATA_ROOT}/BONBID2023_Train/df.csv',
    dimension = '2d',
    transform=resample
)

val_dataset = HIE_Dataset(
    images_dir = [f'{DATA_ROOT}/BONBID2023_Val/2Z_ADC'],
    masks_dir = f'{DATA_ROOT}/BONBID2023_Val/3Label',
    csv_file = f'{DATA_ROOT}/BONBID2023_Val/df.csv',
    dimension = '2d',
    transform=resample
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# %%

header = ["Epoch"]+["Train "+i[0] for i in metrics]+["Train Loss"]+["Val "+i[0] for i in metrics]+["Val Loss"]+["Best Score", "Best Loss"]
stats_df = pd.DataFrame(columns=header)

for epoch in range(START_EPOCH+1,START_EPOCH+NUM_EPOCHS+1):
    print("Epoch:", epoch)
    train_logs = epoch_runner("train", train_loader, model, loss, metrics, optimizer, DEVICE)
    valid_logs = epoch_runner("val", val_loader, model, loss, metrics, device=DEVICE)
    
    best=False
    if valid_logs[metrics[0][0]] >= best_score:
        best_score = valid_logs[metrics[0][0]]
        best = True
    
    if valid_logs['Loss'] <= best_loss:
        best_loss = valid_logs['Loss']
        best = True

    checkpoint = append_metrics_to_df(stats_df, (train_logs, "Train "), (valid_logs, "Val "), ({"Epoch": epoch, "Best Score": best_score, "Best Loss": best_loss}, ""))
    stats_df.to_csv(f"{DEST_DIR}/logs/stats_{START_EPOCH+1}_{ENCODER}.csv", index=False)

    wandb.log(checkpoint)

    checkpoint["model_state_dict"] = model.state_dict()
    checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    checkpoint["batch_size"] = train_loader.batch_size
    checkpoint["encoder"] = ENCODER

    torch.save(checkpoint, f"{DEST_DIR}/models/latest_model_{ENCODER}.pth")
    if best:
        torch.save(checkpoint, f"{DEST_DIR}/models/model_epoch_{epoch}_{ENCODER}.pth")

wandb.finish()




