import torch
import torch.nn as nn
from library.config import Config


class MixupCrossEntropy(nn.Module):
    """
    Custom Cross Entropy Loss that handles Mixup augmentation.
    Computes the weighted sum of losses for the two mixed targets.
    """

    def __init__(self, weight: torch.Tensor = None):
        """
        Args:
            weight (torch.Tensor, optional): A manual rescaling weight given to each class.
                                            If given, has to be a Tensor of size `C`.
        """
        super(MixupCrossEntropy, self).__init__()
        # PyTorch's CrossEntropyLoss supports soft targets (probabilistic labels)
        # and class weights simultaneously.
        self.criterion = nn.CrossEntropyLoss(weight=weight)

    def forward(
        self, preds: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float
    ) -> torch.Tensor:
        """
        Calculates the loss.

        Args:
            preds (torch.Tensor): Model output logits of shape (Batch, Num_Classes).
            y_a (torch.Tensor): First set of targets (original) of shape (Batch, Num_Classes).
            y_b (torch.Tensor): Second set of targets (shuffled) of shape (Batch, Num_Classes).
            lam (float): Mixing coefficient lambda.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Calculate loss for both targets and combine
        loss_a = self.criterion(preds, y_a)
        loss_b = self.criterion(preds, y_b)

        return lam * loss_a + (1 - lam) * loss_b


def get_loss(config: Config, class_weights: torch.Tensor = None) -> nn.Module:
    """
    Factory function to instantiate the MixupCrossEntropy loss.

    Args:
        config (Config): Configuration object containing device settings.
        class_weights (torch.Tensor, optional): Calculated class weights for imbalance handling.

    Returns:
        nn.Module: The instantiated loss function.
    """
    if class_weights is not None:
        class_weights = class_weights.to(config.device)

    return MixupCrossEntropy(weight=class_weights)
