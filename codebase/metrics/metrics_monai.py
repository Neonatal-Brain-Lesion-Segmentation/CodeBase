import torch
import numpy as np
from monai.metrics import DiceMetric, SurfaceDistanceMetric, SurfaceDiceMetric, compute_average_surface_distance, compute_surface_dice

class Dice(DiceMetric):
    """
    A class to safely calculate the Dice coefficient with edge case handling.
    Returns 1.0 if both are empty, and 0.0 if only one is empty.
    """
    def __call__(self, y_pred: torch.Tensor, y: torch.Tensor) -> float:
        if torch.sum(y_pred) == 0 and torch.sum(y) == 0:
            return 1.0
        elif torch.sum(y_pred) == 0 or torch.sum(y) == 0:
            return 0.0
        else:
            return super().__call__(y_pred, y).float()  # Use standard metric otherwise

# class MASD:
#     """
#     A class to safely calculate the Mean Average Surface Distance (MASD) with edge case handling.
#     Returns 0.0 if both are empty, and infinity if only one is empty.
#     """
#     def __call__(self, y_pred: torch.Tensor, y: torch.Tensor) -> float:
#         if torch.sum(y_pred) == 0 and torch.sum(y) == 0:
#             return 0.0
#         elif torch.sum(y_pred) == 0 or torch.sum(y) == 0:
#             return np.sqrt(np.sum(np.array(y.shape)**2))
#         masd = compute_average_surface_distance(y_pred=y_pred, y=y, distance_metric="euclidean")
#         return masd.mean().item()

class MASD(SurfaceDistanceMetric):
    """
    A class to safely calculate the Mean Average Surface Distance (MASD) with edge case handling.
    Returns 0.0 if both are empty, and a max value if only one is empty.
    """
    def __call__(self, y_pred: torch.Tensor, y: torch.Tensor) -> float:
        if torch.sum(y_pred) == 0 and torch.sum(y) == 0:
            return 0.0
        elif torch.sum(y_pred) == 0 or torch.sum(y) == 0:
            return np.sqrt(np.sum(np.array(y.shape)**2))
        else:
            return super().__call__(y_pred, y).float()  # Use standard metric otherwise
    

# class NSD:
#     """
#     A class to safely calculate the Normalized Surface Dice (NSD) with edge case handling.
#     Returns 1.0 if both are empty, and 0.0 if only one is empty.
#     """
#     def __init__(self, dilation: int = 3):
#         self.dilation = dilation

#     def __call__(self, y_pred: torch.Tensor, y: torch.Tensor) -> float:
#         if torch.sum(y_pred) == 0 and torch.sum(y) == 0:
#             return 1.0
#         elif torch.sum(y_pred) == 0 or torch.sum(y) == 0:
#             return 0.0
#         nsd = compute_surface_dice(y_pred=y_pred, y=y, class_thresholds=[self.dilation, self.dilation], distance_metric="euclidean")
#         return nsd.mean().item()

class NSD(SurfaceDiceMetric):
    """
    A class to safely calculate the Normalized Surface Dice (NSD) with edge case handling.
    Returns 1.0 if both are empty, and 0.0 if only one is empty.
    """
    def __call__(self, y_pred, y):
        if torch.sum(y_pred) == 0 and torch.sum(y) == 0:
            return 1.0
        elif torch.sum(y_pred) == 0 or torch.sum(y) == 0:
            return 0.0
        else:
            return super().__call__(y_pred, y).float()  # Use standard metric otherwise


