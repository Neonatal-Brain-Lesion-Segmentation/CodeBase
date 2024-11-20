import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from data_organization import reassemble_to_3d
from transforms.preprocess_v2 import transform_2d_inner, padding
import monai

def make_checkpoint_dir(dest_dir: str) -> None:
    """
    Takes as input a destination directory, and creates the directory and necessary subdirectories for storing models and logs (if not already created).
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    if not os.path.exists(f"{dest_dir}/models"):
        os.makedirs(f"{dest_dir}/models")
    
    if not os.path.exists(f"{dest_dir}/logs"):
        os.makedirs(f"{dest_dir}/logs")

def append_metrics_to_df(df: pd.DataFrame, *args: tuple[dict, str]) -> dict:
    """
    Appends multiple dictionaries with respective prefixes to the DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame to append the metrics to.
        *args (tuple[dict, str]): Tuples of (dictionary, prefix) where the dictionary contains metrics
                                  and the prefix is prepended to the dictionary keys.

    Returns:
        Dictionary containing the combined metrics.
    """
    combined_metrics = {}
    
    for metrics, prefix in args:
        prefixed_metrics = {f"{prefix}{key}": value for key, value in metrics.items()}
        combined_metrics.update(prefixed_metrics)
    
    df.loc[len(df.index)] = combined_metrics

    return combined_metrics

def epoch_runner(description:str, loader:torch.utils.data.DataLoader, model, loss, metrics:list[tuple], optimizer=None, device="cuda", threshold=0.5):

    sample_count = 0

    epoch_metrics = {"Loss": 0.0}
    for metric in metrics:
        epoch_metrics[metric[0]] = 0.0

    run_modes = {"train":True,"val":False}
    mode = run_modes[description.lower()]

    if mode:
        model.train()
    else:
        model.eval()
    
    with torch.set_grad_enabled(mode):
        with tqdm(loader, desc=description.title()) as iterator:
            for images, labels in iterator:
                images = images.to(device)
                labels = labels.to(device)
                
                btch_size = images.shape[0]

                if mode:
                    optimizer.zero_grad()

                outputs = model.forward(images)
                loss_value = loss(outputs, labels)

                epoch_metrics["Loss"] += btch_size*loss_value.item()
                sample_count += btch_size

                predicted = (outputs >= threshold).float()

                for metric in metrics:
                    val = metric[1](predicted, labels)
                    epoch_metrics[metric[0]] += torch.sum(val).item()

                if mode:
                    loss_value.backward()
                    optimizer.step()
                    
                # create postfix dict with avg of what's in epoch metrics dict
                postfix = {k:v/sample_count for k,v in epoch_metrics.items()}
                iterator.set_postfix(postfix)

    return {k:v/sample_count for k,v in epoch_metrics.items()}

def inference_3d_runner(image_paths, label_path, uid_list, modes, model, metrics, device = 'cuda'):
    """
    Have not optimized it to take metrics from the user, will do that soon! For now, default metrics are
    Dice, MASD, and NSD.
    metrics = [('Dice', monai.metrics.DiceMetric(include_background=True,ignore_empty=False))]
    """

    metric_logs = {i[0]:[] for i in metrics}

    preds_3d = {i: [] for i in uid_list}
    masks_3d = {uid: reassemble_to_3d(label_path, uid) for uid in uid_list}

    dice_l = []
    masd_l = []
    nsd_l = []

    with tqdm(uid_list, desc="Val 3D") as pbar:
        for uid in pbar:
            image_set = [reassemble_to_3d(path, uid) for path in image_paths]
            with torch.no_grad():
                for i in range(image_set[0].shape[0]):
                    image = transform_2d_inner(np.stack([image_set[j][i] for j in range(len(image_set))]), modes).unsqueeze(0)
                    # image = np.expand_dims(resample(np.stack([image_set[i]])),axis=0)

                    # image = torch.tensor(image).to(device)
                    image = image.to(device)

                    output = model(image)
                    pred = (output >= 0.5).float()

                    shape = image_set[0].shape
                    shape = (1,shape[1],shape[2])

                    preds_3d[uid].append(padding(pred.cpu().detach().numpy()[0],target_size=tuple(shape))[0])   

                    if len(preds_3d[uid]) == image_set[0].shape[0]:
                        preds_3d[uid] = np.stack(preds_3d[uid])

                        # dice = monai.metrics.DiceMetric(include_background=True,ignore_empty=False)
                        # masd = monai.metrics.SurfaceDistanceMetric(include_background=False, symmetric = True)
                        # nsd = monai.metrics.SurfaceDiceMetric(include_background=False, distance_metric="euclidean", class_thresholds=[2])

                        preds_mask = torch.tensor(preds_3d[uid]).unsqueeze(0).unsqueeze(0)
                        true_mask = torch.tensor(masks_3d[uid]).unsqueeze(0).unsqueeze(0)

                        for metric in metrics:
                            val = metric[1](preds_mask, true_mask)
                            metric_logs[metric[0]].append(val.item())
                        
                        pbar.set_postfix({k:np.mean(v) for k,v in metric_logs.items()})
    
    return {k:np.mean(v) for k,v in metric_logs.items()}

                        # dice_val = dice(preds_mask,true_mask).item()
                        # masd_val = masd(preds_mask,true_mask).item()
                        # nsd_val = nsd(preds_mask,true_mask).item()
                        # dice_l.append(dice_val)
                        # masd_l.append(masd_val)
                        # nsd_l.append(nsd_val)

def resume_checkpoint(dest_dir: str, model, optimizer, device:str, model_dict:str = 'model_state_dict', optimizer_dict = "optimizer_state_dict", epoch:None|int=None, string:str = "", verbose=True, return_list = ["Epoch","Best Score","Best Loss"], return_dict = ["Best 3D Dice","Best 3D MASD","Best 3D NSD"], prefix=False) -> dict:
    """
    Loads a model's and optimzer's state from a checkpoint file. By default, it loads the latest model. 
    However, If an epoch number is present, then it will load the model from that epoch. A string can be added to the model name for better identification.
    Verbose = True will print the loaded checkpoint.
    
    Returns the epoch, best score, and best loss.
    """
    if epoch is not None:
        try:
            checkpoint = torch.load(f"{dest_dir}/model_epoch_{epoch}{string}.pth",map_location=torch.device(device))
        except:
            print("it's a 3d best model")
            checkpoint = torch.load(f"{dest_dir}/model_epoch_{epoch}{string}_3d.pth",map_location=torch.device(device))

    else:
        checkpoint = torch.load(f"{dest_dir}/latest_model{string}.pth",map_location=torch.device(device))
    
    model.load_state_dict(checkpoint[model_dict])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint[optimizer_dict])

    del checkpoint[model_dict]
    del checkpoint[optimizer_dict]

    if verbose:
        print("Loaded checkpoint:")
        for key, value in checkpoint.items():
            print(f"{key}: {value}")
    
    return_d = {}
    for i in return_dict:
        return_d[i.split(' ')[-1] if prefix==False else i] = checkpoint[i]
    
    return tuple([checkpoint[i] for i in return_list]+[return_d])



def epoch_runner_save_image(description: str, loader: torch.utils.data.DataLoader, model, loss, metrics: list[tuple],
                 optimizer=None, device="cuda", threshold=0.5, save_images=False, save_path="saved_images"):
    
    sample_count = 0
    epoch_metrics = {"Loss": 0.0}
    for metric in metrics:
        epoch_metrics[metric[0]] = 0.0

    run_modes = {"train": True, "val": False}
    mode = run_modes[description.lower()]
    if mode:
        model.train()
    else:
        model.eval()

    # Ensure the save directory exists if saving images
    if save_images:
        os.makedirs(save_path, exist_ok=True)

    with torch.set_grad_enabled(mode):
        with tqdm(loader, desc=description.title()) as iterator:
            for idx, (images, labels) in enumerate(iterator):
                images = images.to(device)
                labels = labels.to(device)
                
                btch_size = images.shape[0]
                if mode:
                    optimizer.zero_grad()

                outputs = model.forward(images)
                loss_value = loss(outputs, labels)

                epoch_metrics["Loss"] += btch_size * loss_value.item()
                sample_count += btch_size

                predicted = (outputs >= threshold).float()

                for metric in metrics:
                    val = metric[1](predicted, labels)
                    epoch_metrics[metric[0]] += torch.sum(val).item()

                if mode:
                    loss_value.backward()
                    optimizer.step()

                # Save images with annotations if enabled
                if save_images:
                    for i in range(btch_size):
                        # Prepare the image, GT, and PR
                        img = images[i].detach().cpu().numpy()
                        
                        gt = labels[i].detach().cpu().numpy()
                        pr = predicted[i].detach().cpu().numpy()

                        # Plot using Matplotlib
                        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                        axes[0].imshow(img.transpose(1, 2, 0) if img.shape[0] > 1 else img[0], cmap="gray")
                        axes[0].set_title("Original")
                        axes[0].axis("off")

                        axes[1].imshow(gt, cmap="gray")
                        axes[1].set_title("Ground Truth")
                        axes[1].axis("off")

                        axes[2].imshow(pr, cmap="gray")
                        axes[2].set_title("Prediction")
                        axes[2].axis("off")

                        # Add Loss and Metric as a Title
                        metric_name = metrics[0][0]
                        metric_value = torch.sum(val).item()
                        fig.suptitle(f"Loss: {loss_value.item():.4f}, {metric_name}: {metric_value:.4f}", fontsize=12)

                        # Save the plot
                        save_name = f"{save_path}/{description.lower()}_{idx}_{i}.png"
                        plt.savefig(save_name, bbox_inches="tight")
                        plt.close(fig)

                # Create postfix dict with averages
                postfix = {k: v / sample_count for k, v in epoch_metrics.items()}
                iterator.set_postfix(postfix)

    return epoch_metrics

