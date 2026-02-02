import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def seed_everything(seed: int):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_rmse(predictions, targets):
    """
    Calculates the Root Mean Squared Error (RMSE) between predictions and targets.

    Args:
        predictions (torch.Tensor or np.ndarray): Predicted pixel intensities.
        targets (torch.Tensor or np.ndarray): Ground truth pixel intensities.

    Returns:
        float: The computed RMSE value.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Ensure float type for precision
    predictions = predictions.astype(np.float32)
    targets = targets.astype(np.float32)

    # Compute MSE and then RMSE
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)

    return float(rmse)


class MultiScaleMSELoss(nn.Module):
    """
    Custom loss module that computes the weighted sum of MSE losses from the
    main output and auxiliary deep supervision heads.
    """

    def __init__(self, weights=None):
        """
        Args:
            weights (list[float], optional): A list of weights for the outputs.
                The first weight applies to the final output, subsequent weights
                apply to auxiliary outputs. If None, all outputs are weighted equally (1.0).
        """
        super(MultiScaleMSELoss, self).__init__()
        self.weights = weights
        self.mse = nn.MSELoss()

    def forward(self, predictions, target):
        """
        Computes the multi-scale MSE loss.

        Args:
            predictions (torch.Tensor or list[torch.Tensor]): The model output.
                Can be a single tensor or a list of tensors (deep supervision).
            target (torch.Tensor): The ground truth image.

        Returns:
            torch.Tensor: The calculated loss.
        """
        # Case 1: Single output (Validation or Deep Supervision disabled)
        if torch.is_tensor(predictions):
            return self.mse(predictions, target)

        # Case 2: List of outputs (Deep Supervision enabled)
        if isinstance(predictions, (list, tuple)):
            total_loss = 0.0
            num_preds = len(predictions)

            # Use provided weights or default to 1.0 for all
            if self.weights is not None:
                current_weights = self.weights
                # Pad weights with 1.0 if not enough provided
                if len(current_weights) < num_preds:
                    current_weights = current_weights + [1.0] * (
                        num_preds - len(current_weights)
                    )
            else:
                current_weights = [1.0] * num_preds

            for i, pred in enumerate(predictions):
                # Interpolate prediction to match target spatial dimensions
                if pred.shape[-2:] != target.shape[-2:]:
                    pred_interpolated = F.interpolate(
                        pred,
                        size=target.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                else:
                    pred_interpolated = pred

                # Calculate MSE for this scale
                scale_loss = self.mse(pred_interpolated, target)
                total_loss += current_weights[i] * scale_loss

            return total_loss

        # Fallback for unexpected types
        return self.mse(predictions, target)
