import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

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

def resume_checkpoint(dest_dir: str, model, optimizer, device:str, model_dict:str = 'model_state_dict', optimizer_dict = "optimizer_state_dict", epoch:None|int=None, string:str = "", verbose=True) -> dict:
    """
    Loads a model's and optimzer's state from a checkpoint file. By default, it loads the latest model. 
    However, If an epoch number is present, then it will load the model from that epoch. A string can be added to the model name for better identification.
    Verbose = True will print the loaded checkpoint.
    
    Returns the epoch, best score, and best loss.
    """
    if epoch is not None:
        checkpoint = torch.load(f"{dest_dir}/model_epoch_{epoch}{string}.pth",map_location=torch.device(device))
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

    return checkpoint["Epoch"], checkpoint["Best Score"], checkpoint["Best Loss"]


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
