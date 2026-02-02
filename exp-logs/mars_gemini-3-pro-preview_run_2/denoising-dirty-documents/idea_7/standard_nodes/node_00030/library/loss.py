import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleMSELoss(nn.Module):
    """
    Computes the Multi-Scale Mean Squared Error Loss for Deep Supervision.

    The model is trained to predict the noise residual (Noisy Input - Clean Label).
    This loss function calculates the MSE between the predicted noise and the
    actual noise. It handles multiple outputs from the model (Deep Supervision)
    by comparing each output to the ground truth noise, dynamically resizing
    the ground truth if necessary to match the resolution of auxiliary heads.
    """

    def __init__(self, weights=None):
        """
        Args:
            weights (list of float, optional): A list of weights for each output scale.
                                               The first weight corresponds to the final output,
                                               subsequent weights to auxiliary outputs.
                                               If None, defaults to 1.0 for all scales.
        """
        super(MultiScaleMSELoss, self).__init__()
        self.weights = weights
        self.mse = nn.MSELoss()

    def forward(self, preds, noisy_imgs, clean_imgs):
        """
        Args:
            preds (list[torch.Tensor] or torch.Tensor): The predictions from the model.
                If Deep Supervision is enabled, this is a list where the first element
                is the final output and subsequent elements are auxiliary outputs.
            noisy_imgs (torch.Tensor): The input noisy images (B, C, H, W).
            clean_imgs (torch.Tensor): The ground truth clean images (B, C, H, W).

        Returns:
            torch.Tensor: The calculated scalar loss.
        """
        # Calculate the ground truth noise residual
        # Target Noise = Input (Noisy) - Label (Clean)
        target_noise = noisy_imgs - clean_imgs

        # Ensure preds is a list to handle both single-output and multi-output cases uniformly
        if isinstance(preds, torch.Tensor):
            preds = [preds]

        # Determine weights for the loss terms
        if self.weights is None:
            # Default to equal weighting (1.0) if no weights provided
            weights = [1.0] * len(preds)
        elif len(self.weights) != len(preds):
            # If provided weights don't match number of outputs, fallback to 1.0
            # This prevents index errors if model architecture changes
            weights = [1.0] * len(preds)
        else:
            weights = self.weights

        total_loss = 0.0

        for i, pred in enumerate(preds):
            # Dynamic Resolution Matching:
            # Check if the prediction spatial dimensions match the target.
            # If not (e.g., auxiliary head outputs lower resolution),
            # downsample the ground truth target to match the prediction.
            if pred.shape[2:] != target_noise.shape[2:]:
                target_resampled = F.interpolate(
                    target_noise,
                    size=pred.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                target_resampled = target_noise

            # Compute MSE for this scale
            scale_loss = self.mse(pred, target_resampled)

            # Accumulate weighted loss
            total_loss += weights[i] * scale_loss

        return total_loss
