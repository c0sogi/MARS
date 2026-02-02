import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from library.config import Config


class WeightedBCELoss(nn.Module):
    """
    Binary Cross Entropy Loss with Logits and optional positive class weighting.
    Handles class imbalance by weighting the positive class.
    """

    def __init__(self, pos_weight=None):
        """
        Args:
            pos_weight (torch.Tensor or float, optional): Weight for the positive class.
                                                          If float, converted to Tensor.
        """
        super(WeightedBCELoss, self).__init__()

        if pos_weight is not None:
            if not isinstance(pos_weight, torch.Tensor):
                pos_weight = torch.tensor([pos_weight], dtype=torch.float32)

        # Register as buffer so it moves to device with the module
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, input, target):
        """
        Args:
            input (torch.Tensor): Logits from the model (B, 1) or (B,)
            target (torch.Tensor): Ground truth labels (B, 1) or (B,)
        """
        # Ensure target shape matches input shape
        if target.shape != input.shape:
            target = target.view_as(input)

        target = target.float()

        return F.binary_cross_entropy_with_logits(
            input, target, pos_weight=self.pos_weight, reduction="mean"
        )


class MixupLoss(nn.Module):
    """
    Wrapper for calculating Mixup loss.
    Computes loss as: lambda * loss(pred, y_a) + (1 - lambda) * loss(pred, y_b)
    """

    def __init__(self, criterion):
        """
        Args:
            criterion (nn.Module): The base loss function (e.g., WeightedBCELoss).
        """
        super(MixupLoss, self).__init__()
        self.criterion = criterion

    def forward(self, preds, y_a, y_b, lam):
        """
        Args:
            preds (torch.Tensor): Model predictions.
            y_a (torch.Tensor): Targets for the first image.
            y_b (torch.Tensor): Targets for the second image.
            lam (float): The mixup lambda coefficient.
        """
        loss_a = self.criterion(preds, y_a)
        loss_b = self.criterion(preds, y_b)

        return lam * loss_a + (1 - lam) * loss_b


def get_loss_module():
    """
    Factory function to initialize the loss module.
    Calculates positive class weight from training metadata if configured.

    Returns:
        nn.Module: Configured WeightedBCELoss instance.
    """
    pos_weight = None

    if Config.USE_CLASS_WEIGHTS:
        try:
            # Load training metadata
            train_df = pd.read_csv(Config.TRAIN_CSV)

            # Calculate class distribution
            counts = train_df["label"].value_counts()
            neg_count = counts.get(0, 0)
            pos_count = counts.get(1, 0)

            if pos_count > 0:
                # Calculate inverse class frequency weight
                # weight = number_of_negatives / number_of_positives
                weight_val = neg_count / pos_count
                pos_weight = torch.tensor([weight_val], dtype=torch.float32)
                print(
                    f"Initializing WeightedBCELoss. Positive Class Weight: {weight_val:.4f} (Neg: {neg_count}, Pos: {pos_count})"
                )
            else:
                print(
                    "Warning: No positive samples in training data. Defaulting to weight 1.0."
                )

        except FileNotFoundError:
            print(
                f"Warning: Metadata file not found at {Config.TRAIN_CSV}. Cannot calculate weights."
            )
        except Exception as e:
            print(f"Error calculating class weights: {e}. Defaulting to weight 1.0.")

    return WeightedBCELoss(pos_weight=pos_weight)
