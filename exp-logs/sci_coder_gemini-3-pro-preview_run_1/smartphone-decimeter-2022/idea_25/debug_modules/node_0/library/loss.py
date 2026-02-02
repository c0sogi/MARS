import torch
import torch.nn as nn
from typing import List, Union, Dict, Optional


class DecimatedDeepSupervisionLoss(nn.Module):
    """
    Implements Mean Absolute Error (MAE) loss with Decimated Deep Supervision.

    This loss function expects a list of model predictions at decreasing temporal resolutions
    (e.g., full, 1/2, 1/4, 1/8). It compares each prediction against the ground truth
    decimated (subsampled) by the corresponding factor.

    Args:
        weights (List[float], optional): A list of weights for each scale's loss.
                                         If None, all scales are weighted equally (1.0).
    """

    def __init__(self, weights: Optional[List[float]] = None):
        super(DecimatedDeepSupervisionLoss, self).__init__()
        self.criterion = nn.L1Loss(reduction="mean")
        self.weights = weights

    def forward(
        self,
        preds: List[torch.Tensor],
        targets: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Calculates the weighted sum of MAE losses across all scales.

        Args:
            preds: List of tensors [pred_scale_0, pred_scale_1, ...].
                   pred_scale_0 is the final high-res output.
                   Subsequent elements are auxiliary outputs at lower resolutions.
            targets: Ground truth target. Can be a single high-resolution Tensor
                     or a Dictionary containing 'scale_0' (high-res target).

        Returns:
            torch.Tensor: The weighted sum of losses.
        """
        # 1. Extract High-Resolution Ground Truth
        if isinstance(targets, dict):
            # If dataset returns a dict, use 'scale_0' as the reference high-res target
            gt = targets.get("scale_0")
            if gt is None:
                raise KeyError(
                    "Target dictionary must contain key 'scale_0' for high-resolution ground truth."
                )
        else:
            gt = targets

        # 2. Initialize Weights if not provided
        if self.weights is None:
            # Default to equal weighting (1.0) for all scales if not specified
            weights = [1.0] * len(preds)
        else:
            if len(self.weights) != len(preds):
                raise ValueError(
                    f"Length of weights ({len(self.weights)}) must match number of predictions ({len(preds)})."
                )
            weights = self.weights

        total_loss = 0.0

        # 3. Compute Loss per Scale
        for i, pred in enumerate(preds):
            # Calculate decimation factor: 2^i (1, 2, 4, 8, ...)
            factor = 2**i

            # Decimate ground truth to match prediction resolution
            # We use slicing [..., ::factor] which selects every factor-th element
            # This aligns with the MaxPool1d(2) operations in the encoder
            target_decimated = gt[..., ::factor]

            # Handle potential length mismatches (e.g. due to padding or odd lengths)
            # We truncate to the minimum length between prediction and target
            len_pred = pred.shape[-1]
            len_target = target_decimated.shape[-1]

            if len_pred != len_target:
                min_len = min(len_pred, len_target)
                pred_sliced = pred[..., :min_len]
                target_sliced = target_decimated[..., :min_len]
            else:
                pred_sliced = pred
                target_sliced = target_decimated

            # Calculate MAE for this scale
            scale_loss = self.criterion(pred_sliced, target_sliced)

            # Add weighted loss
            total_loss += weights[i] * scale_loss

        return total_loss
