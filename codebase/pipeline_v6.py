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
from mix_loss import TverskyLogHausdorffFocalLoss


import os
import wandb

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchio")

# %%
WANDB_ON = True
RESUME = 1
RESUME_EPOCH = None
NUM_EPOCHS = 30
num_channels = 3
mode = '2D'
# ENCODER = "densenet161"
ENCODER = "inceptionresnetv2"
# ENCODER = "inceptionv4"
# ENCODER = "transunet"
BATCH_SIZE = 16


DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
print(DEVICE)

DEST_DIR = f"/home/lakes/bonbid-2024/checkpoints/UPP-3C-{ENCODER}-TLHF-DD-NewSplit"
make_checkpoint_dir(DEST_DIR)   

DATA_ROOT = "/home/lakes/bonbid-2024/Dataset"
print(os.getcwd())
print(os.listdir(DATA_ROOT))

if WANDB_ON:
    wandb.init(project=f"UPP-3C-{ENCODER}-TLHF-DD-NewSplit",
            name = "Run-1",
                config={
            "mode": mode,
            "learning_rate": 0.0001,
            "architecture": ENCODER,
            "dataset": "BONBID-2024",
            "epochs": NUM_EPOCHS,
            # "LR Scheduler":"Cosine Annealing",
            "Message":"Changing LRs",
            "Loss":"2*Focal+2*H+1.5*T",
            "Loss Info":"AlphaT=0.3, BetaT=0.7, GammaF=3"
        },)

# %%

model = smp.UnetPlusPlus(
            encoder_name=ENCODER,
            encoder_weights=None,
            in_channels=num_channels,
            classes=1,
            activation="sigmoid",
            decoder_attention_type = "scse"
        )
# %%


# initialize_weights(model)

model.to(DEVICE)

# loss = monai.losses.DiceFocalLoss()
# loss = TverskyHausdorffFocalLoss(alpha_focal=0.25, gamma_focal=2.0, lambda_tversky=1, lambda_focal=1, alpha_tversky=0.3, beta_tversky=0.7, lambda_hausdorff=1, alpha_hausdorff=2)
loss = TverskyLogHausdorffFocalLoss(lambda_focal=2, lambda_hausdorff=2, lambda_tversky=1.5, alpha_tversky=0.3, beta_tversky=0.7, gamma_focal=3)

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

# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#     optimizer = optimizer,
#     T_max = 20,
#     verbose = True
# )

scheduler = None

best_score = 0
best_loss = np.inf
START_EPOCH = -1

if RESUME:
    START_EPOCH, best_score, best_loss, metrics_best_3d_dict = resume_checkpoint(f"{DEST_DIR}/models",model,optimizer,scheduler=scheduler, device=DEVICE,epoch=RESUME_EPOCH,string=f"_{ENCODER}",return_list=["Epoch","Best Score","Best Loss"],return_dict=["Best 3D Dice","Best 3D MASD","Best 3D NSD"])


optimizer.param_groups[0]["lr"] = 0.7*0.0001
# print("LR:",optimizer.param_groups[0]["lr"])
# %%
train_dataset = HIE_Dataset(
    images_dir = [f'{DATA_ROOT}/BONBID2024_Train/ADC',f'{DATA_ROOT}/BONBID2024_Train/Z_ADC'] if num_channels == 2 else [f'{DATA_ROOT}/Z_ADC',f'{DATA_ROOT}/Z_ADC',f'{DATA_ROOT}/ADC'],
    masks_dir = f'{DATA_ROOT}/LABEL',
    csv_file = f'{DATA_ROOT}/metadata_train.csv',
    dimension = mode,
    transform=transform_2d,
    augment=augment,
    mode=["ZADC","ZADC_Clip","ADC"]
)

val_dataset = HIE_Dataset(
    images_dir = [f'{DATA_ROOT}/BONBID2024_Val/ADC',f'{DATA_ROOT}/BONBID2024_Val/Z_ADC'] if num_channels == 2 else [f'{DATA_ROOT}/Z_ADC',f'{DATA_ROOT}/Z_ADC',f'{DATA_ROOT}/ADC'],
    masks_dir = f'{DATA_ROOT}/LABEL',
    csv_file = f'{DATA_ROOT}/metadata_val.csv',
    dimension = mode,
    transform=transform_2d,
    mode=["ZADC","ZADC_Clip","ADC"]
)

img = train_dataset[33][0]
mask = train_dataset[33][1]
print(mask.shape,mask.min(),mask.max())
print(img.shape,img[0].min(),img[0].max(),'\n',img[1].min(),img[1].max(),'\n',img[2].min(),img[2].max())

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,num_workers=2)

# %%

header = ["Epoch"]+["Train "+i[0] for i in metrics]+["Train Loss"]+["Val "+i[0] for i in metrics]+["Val Loss"]+["Best Score", "Best Loss","LR"]+["Val 3D "+i[0] for i in metrics_3d]+["Best 3D "+i[0] for i in metrics_3d]
stats_df = pd.DataFrame(columns=header)

for epoch in range(START_EPOCH+1,START_EPOCH+NUM_EPOCHS+1):
    print("Epoch:", epoch, "| LR:",optimizer.param_groups[0]["lr"])
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

    if scheduler is not None:
        scheduler.step()
    
    if WANDB_ON:
        wandb.log(checkpoint)

    checkpoint["model_state_dict"] = model.state_dict()
    checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    checkpoint["batch_size"] = train_loader.batch_size
    checkpoint["encoder"] = ENCODER

    torch.save(checkpoint, f"{DEST_DIR}/models/latest_model_{ENCODER}.pth")
    if best or best_3d:
        print(f"A best model{best_3d if best_3d else ''} has been saved!")
        torch.save(checkpoint, f"{DEST_DIR}/models/model_epoch_{epoch}_{ENCODER}{best_3d if best_3d else ''}.pth")
    
    # if epoch == 149:
    #     optimizer.param_groups[0]["lr"] = 0.4*0.0001
    # if epoch == 199:
    #     optimizer.param_groups[0]["lr"] = 0.1*0.0001
    # if epoch == 249:
    #     optimizer.param_groups[0]["lr"] = 0.7*0.1*0.0001

if WANDB_ON:
    wandb.finish()