import torch
import torch.nn as nn
from library.utils import dice_coef


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation tasks.
    Calculates 1 - Dice Coefficient.
    """

    def __init__(self, smooth: float = 1.0, from_logits: bool = True):
        """
        Args:
            smooth (float): Smoothing factor to prevent division by zero.
            from_logits (bool): If True, applies sigmoid to inputs.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred (torch.Tensor): Predicted logits or probabilities.
            y_true (torch.Tensor): Ground truth binary mask.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        if self.from_logits:
            y_pred = torch.sigmoid(y_pred)

        # dice_coef expects (y_true, y_pred)
        dice_score = dice_coef(y_true, y_pred, smooth=self.smooth)
        return 1.0 - dice_score


class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross Entropy and Dice Loss.
    Useful for segmentation tasks to balance pixel-wise accuracy and overlap.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1.0,
    ):
        """
        Args:
            bce_weight (float): Weight for BCE loss.
            dice_weight (float): Weight for Dice loss.
            smooth (float): Smoothing factor for Dice loss.
        """
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth, from_logits=True)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred (torch.Tensor): Predicted logits (B, C, H, W).
            y_true (torch.Tensor): Ground truth binary mask (B, C, H, W).

        Returns:
            torch.Tensor: Weighted combined loss.
        """
        bce_loss = self.bce(y_pred, y_true)
        dice_loss = self.dice(y_pred, y_true)

        return (self.bce_weight * bce_loss) + (self.dice_weight * dice_loss)
