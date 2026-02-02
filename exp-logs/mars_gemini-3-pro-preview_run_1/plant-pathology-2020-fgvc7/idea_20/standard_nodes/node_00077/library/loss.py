import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedSoftTargetCrossEntropy(nn.Module):
    """
    Weighted Cross Entropy Loss that accepts soft targets (probabilities).

    Formula:
        Loss = - (1/N) * sum_i ( sum_c ( weight_c * target_i,c * log(predicted_i,c) ) )

    Args:
        weight (torch.Tensor, optional): A tensor of size (C,) containing the weight for each class.
    """

    def __init__(self, weight=None):
        super(WeightedSoftTargetCrossEntropy, self).__init__()
        # Register weight as a buffer so it automatically moves to the correct device
        # when .to(device) is called on the module.
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, input, target):
        """
        Forward pass of the loss function.

        Args:
            input (torch.Tensor): Logits from the model of shape (batch_size, num_classes).
            target (torch.Tensor): Soft targets (probabilities) of shape (batch_size, num_classes).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Compute log probabilities (numerically stable)
        log_probs = F.log_softmax(input, dim=1)

        # Compute element-wise cross entropy: - target * log(prediction)
        loss = -target * log_probs

        # Apply class weights if provided
        if self.weight is not None:
            # self.weight is (num_classes,)
            # loss is (batch_size, num_classes)
            # Broadcasting applies weight to each class column
            loss = loss * self.weight

        # Sum over classes (dim 1) to get loss per sample
        loss = loss.sum(dim=1)

        # Normalize by sum of weights in the batch (Cite Lesson 00076)
        if self.weight is not None:
            # For soft targets, the effective weight of a sample is sum(target * weight)
            sample_weights = (target * self.weight).sum(dim=1)
            return loss.sum() / (sample_weights.sum() + 1e-6)
        else:
            return loss.mean()
