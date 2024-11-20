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
# TODO: Make it such that the user can keep a track of the best scores for all metrics (is a lower score better or a higher score better?)
# and the script should save that model. For NSD new highest score is good, for MASD, new lowest score is good.

# %%

RESUME = 0
RESUME_EPOCH = None
NUM_EPOCHS = 5
mode = '2D'
ENCODER = "densenet121"
BATCH_SIZE = 16


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
    START_EPOCH, best_score, best_loss = resume_checkpoint(f"{DEST_DIR}/models",model,optimizer,DEVICE,epoch=RESUME_EPOCH,string=f"_{ENCODER}")


# %%
train_dataset = HIE_Dataset(
    images_dir = [f'{DATA_ROOT}/BONBID2023_Train/2Z_ADC'],
    masks_dir = f'{DATA_ROOT}/BONBID2023_Train/3Label',
    csv_file = f'{DATA_ROOT}/BONBID2023_Train/df.csv',
    dimension = mode,
    transform=resample
)

val_dataset = HIE_Dataset(
    images_dir = [f'{DATA_ROOT}/BONBID2023_Val/2Z_ADC'],
    masks_dir = f'{DATA_ROOT}/BONBID2023_Val/3Label',
    csv_file = f'{DATA_ROOT}/BONBID2023_Val/df.csv',
    dimension = mode,
    transform=resample
)


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# %%

header = ["Epoch"]+["Train "+i[0] for i in metrics]+["Train Loss"]+["Val "+i[0] for i in metrics]+["Val Loss"]+["Best Score", "Best Loss"]+["Dice 3D", "MASD 3D", "NSD 3D"]
stats_df = pd.DataFrame(columns=header)

for epoch in range(START_EPOCH+1,START_EPOCH+NUM_EPOCHS+1):
    print("Epoch:", epoch)
    train_logs = epoch_runner("train", train_loader, model, loss, metrics, optimizer, DEVICE)
    valid_logs = epoch_runner("val", val_loader, model, loss, metrics, device=DEVICE)
    dice, masd, nsd = inference_3d_runner(val_dataset.images_dir[0],val_dataset.masks_dir,val_dataset.ids,model,DEVICE)
    
    best=False
    if valid_logs[metrics[0][0]] >= best_score:
        best_score = valid_logs[metrics[0][0]]
        best = True
    
    if valid_logs['Loss'] <= best_loss:
        best_loss = valid_logs['Loss']
        best = True

    checkpoint = append_metrics_to_df(stats_df, (train_logs, "Train "), (valid_logs, "Val "), ({"Epoch": epoch, "Best Score": best_score, "Best Loss": best_loss, "Dice 3D":dice, "MASD 3D":masd, "NSD 3D":nsd}, ""))
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




