import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchDiceLoss(nn.Module):
    """
    Computes the Dice Coefficient Loss over the entire batch.

    Instead of computing Dice per sample and averaging, this treats the
    entire batch as a single volume. This stabilizes the gradient,
    especially when the batch size is small or the positive class is very sparse.

    Formula:
        Dice = (2 * |X n Y|) / (|X| + |Y|)
        Loss = 1 - Dice
    """

    def __init__(self, smooth: float = 1e-6):
        """
        Args:
            smooth (float): Smoothing factor to avoid division by zero.
        """
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): Raw model outputs (logits) of shape (N, C, H, W).
            targets (torch.Tensor): Ground truth masks of shape (N, C, H, W).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Apply sigmoid to convert logits to probabilities
        probs = torch.sigmoid(logits)

        # Flatten the tensors to 1D vectors (N*C*H*W)
        # This aggregates statistics over the entire batch
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        # Calculate intersection and union
        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()

        # Compute Dice Score
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1.0 - dice


class DeepSupervisionLoss(nn.Module):
    """
    Hybrid Loss function supporting Deep Supervision.

    Combines Binary Cross Entropy (BCE) and Batch Dice Loss.
    If the model returns a list of outputs (Deep Supervision), it computes
    a weighted sum of losses for each head.
    """

    def __init__(self, config):
        """
        Args:
            config: Configuration object containing loss_weights.
        """
        super().__init__()
        self.config = config
        self.weights = config.loss_weights  # e.g., [1.0, 0.4, 0.2]

        # Components
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = BatchDiceLoss()

    def forward(self, preds, targets):
        """
        Args:
            preds (torch.Tensor or list[torch.Tensor]): Model predictions.
                - List of tensors during training (Main, Aux1, Aux2).
                - Single tensor during validation/testing.
            targets (torch.Tensor): Ground truth masks.

        Returns:
            torch.Tensor: Weighted scalar loss.
        """
        # Case 1: Deep Supervision (List of outputs)
        if isinstance(preds, list):
            total_loss = 0.0

            # Iterate over outputs and corresponding weights
            # We assume the order matches config.loss_weights: [Main, Aux1, Aux2]
            for i, pred in enumerate(preds):
                # Safety check for weights index
                weight = self.weights[i] if i < len(self.weights) else 0.0

                if weight > 0:
                    # Calculate Hybrid Loss for this head
                    loss_bce = self.bce(pred, targets)
                    loss_dice = self.dice(pred, targets)
                    term_loss = loss_bce + loss_dice

                    # Add weighted term
                    total_loss += weight * term_loss

            return total_loss

        # Case 2: Single Output (Validation / Inference)
        else:
            loss_bce = self.bce(preds, targets)
            loss_dice = self.dice(preds, targets)
            return loss_bce + loss_dice
