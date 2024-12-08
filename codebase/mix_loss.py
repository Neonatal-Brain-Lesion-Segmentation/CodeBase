# Tversky + LogHausdorff + Focal Loss
from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.nn as nn

from torch.nn.modules.loss import _Loss

from monai.losses.tversky import TverskyLoss
from monai.losses.focal_loss import FocalLoss
from monai.losses.hausdorff_loss import LogHausdorffDTLoss
from monai.losses.dice import DiceLoss
from monai.networks import one_hot

class DiceBCELoss(_Loss):
    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Callable | None = None,
        reduction: str = "mean",
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
        weight: int|None = None,
        jaccard: bool=False,
        squared_pred:bool = False,
        lambda_dice:int = 1,
        lambda_bce:int = 1
    ) -> None:
        
        super().__init__() 
        self.bce = nn.BCELoss()
        self.dice = DiceLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            softmax=softmax,
            other_act=other_act,
            squared_pred=squared_pred,
            jaccard=jaccard,
            reduction=reduction,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            batch=batch,
            weight=weight
        )

        if lambda_dice < 0.0:
            raise ValueError("lambda_tversky should be no less than 0.0.")
        if lambda_bce < 0.0:
            raise ValueError("lambda_focal should be no less than 0.0.")

        self.lambda_dice = lambda_dice
        self.lambda_bce = lambda_bce
        self.to_onehot_y = to_onehot_y

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
    Args:
        input: the shape should be BNH[WD]. The input should be the original logits
            due to the restriction of ``monai.losses.FocalLoss``.
        target: the shape should be BNH[WD] or B1H[WD].

    Raises:
        ValueError: When number of dimensions for input and target are different.
        ValueError: When number of channels for target is neither 1 (without one-hot encoding) nor the same as input.

    Returns:
        torch.Tensor: value of the loss.
    """
        if input.dim() != target.dim():
            raise ValueError(
                "the number of dimensions for input and target should be the same, "
                f"got shape {input.shape} (nb dims: {len(input.shape)}) and {target.shape} (nb dims: {len(target.shape)}). "
                "if target is not one-hot encoded, please provide a tensor with shape B1H[WD]."
            )

        if target.shape[1] != 1 and target.shape[1] != input.shape[1]:
            raise ValueError(
                "number of channels for target is neither 1 (without one-hot encoding) nor the same as input, "
                f"got shape {input.shape} and {target.shape}."
            )

        if self.to_onehot_y:
            n_pred_ch = input.shape[1]
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)
        
        dice_loss = self.dice(input, target)
        bce_loss = self.bce(input, target)
    
        total_loss: torch.Tensor = self.lambda_dice*dice_loss + self.lambda_bce*bce_loss
        return total_loss


class TverskyLogHausdorffFocalLoss(_Loss):
    """
    Compute both Dice loss and Focal Loss, and return the weighted sum of these two losses.
    The details of Dice loss is shown in ``monai.losses.DiceLoss``.
    The details of Focal Loss is shown in ``monai.losses.FocalLoss``.

    ``gamma`` and ``lambda_focal`` are only used for the focal loss.
    ``include_background``, ``weight``, ``reduction``, and ``alpha`` are used for both losses,
    and other parameters are only used for dice loss.

    """

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Callable | None = None,
        reduction: str = "mean",
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
        gamma_focal: float = 2.0,
        weight: Sequence[float] | float | int | torch.Tensor | None = None,
        lambda_tversky: float = 1.0,
        lambda_focal: float = 1.0,
        lambda_hausdorff: float = 1.0,
        alpha_focal: float | None = None,
        alpha_tversky: float | None = 0.5,
        beta_tversky: float | None = 0.5,
        alpha_hausdorff: float | None = 2.0,
    ) -> None:
        
        super().__init__()
        self.tversky = TverskyLoss(
            include_background=include_background,
            to_onehot_y=False,
            sigmoid=sigmoid,
            softmax=softmax,
            other_act=other_act,
            alpha=alpha_tversky,
            beta=beta_tversky,
            reduction=reduction,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            batch=batch,
        )

        self.focal = FocalLoss(
            include_background=include_background,
            to_onehot_y=False,
            gamma=gamma_focal,
            weight=weight,
            alpha=alpha_focal,
            reduction=reduction,
        )

        self.hausdorff = LogHausdorffDTLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            reduction=reduction,
            sigmoid=sigmoid,
            softmax=softmax,
            other_act=other_act,
            batch=batch,
            alpha=alpha_hausdorff,
        )


        if lambda_tversky < 0.0:
            raise ValueError("lambda_tversky should be no less than 0.0.")
        if lambda_focal < 0.0:
            raise ValueError("lambda_focal should be no less than 0.0.")
        if lambda_hausdorff < 0.0:
            raise ValueError("lambda_hausdorff should be no less than 0.0.")
        self.lambda_tversky = lambda_tversky
        self.lambda_focal = lambda_focal
        self.lambda_hausdorff = lambda_hausdorff
        self.to_onehot_y = to_onehot_y

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD]. The input should be the original logits
                due to the restriction of ``monai.losses.FocalLoss``.
            target: the shape should be BNH[WD] or B1H[WD].

        Raises:
            ValueError: When number of dimensions for input and target are different.
            ValueError: When number of channels for target is neither 1 (without one-hot encoding) nor the same as input.

        Returns:
            torch.Tensor: value of the loss.
        """
        if input.dim() != target.dim():
            raise ValueError(
                "the number of dimensions for input and target should be the same, "
                f"got shape {input.shape} (nb dims: {len(input.shape)}) and {target.shape} (nb dims: {len(target.shape)}). "
                "if target is not one-hot encoded, please provide a tensor with shape B1H[WD]."
            )

        if target.shape[1] != 1 and target.shape[1] != input.shape[1]:
            raise ValueError(
                "number of channels for target is neither 1 (without one-hot encoding) nor the same as input, "
                f"got shape {input.shape} and {target.shape}."
            )

        if self.to_onehot_y:
            n_pred_ch = input.shape[1]
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)
        tversky_loss = self.tversky(input, target)
        focal_loss = self.focal(input, target)
        hausdorff_loss = self.hausdorff(input, target)
        total_loss: torch.Tensor = self.lambda_tversky * tversky_loss + self.lambda_focal * focal_loss + self.lambda_hausdorff * hausdorff_loss
        return total_loss