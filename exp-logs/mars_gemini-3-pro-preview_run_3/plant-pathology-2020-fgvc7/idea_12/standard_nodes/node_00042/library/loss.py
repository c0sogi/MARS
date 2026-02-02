import torch
import torch.nn as nn
from library.config import Config


class DeepSupervisionLoss(nn.Module):
    """
    Implements Cross-Entropy Loss with Deep Supervision support for the Feature Pyramid Network.

    Logic:
    - If the model returns a tuple (training mode), it computes a weighted sum of losses
      from the main head (P3) and auxiliary heads (P4, P5).
    - If the model returns a tensor (validation/inference mode), it computes the loss
      only on the main output.
    """

    def __init__(self, class_weights=None, aux_weight=None):
        """
        Args:
            class_weights (torch.Tensor, optional): Pre-computed inverse frequency weights
                                                    to handle class imbalance.
            aux_weight (float, optional): Weighting factor for auxiliary losses.
                                          Defaults to Config.aux_loss_weight.
        """
        super(DeepSupervisionLoss, self).__init__()

        self.aux_weight = (
            aux_weight if aux_weight is not None else Config.aux_loss_weight
        )

        # Initialize CrossEntropyLoss
        # We use the 'weight' parameter to handle class imbalance.
        # PyTorch's CrossEntropyLoss supports (N, C) float targets (soft labels/one-hot),
        # which matches the output format of our AppleDataset.
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, outputs, targets):
        """
        Computes the Deep Supervision loss.

        Args:
            outputs: Model output.
                     - If Tuple: (logits_p3, logits_p4, logits_p5) [Training]
                     - If Tensor: logits_p3 [Validation/Inference]
            targets: Ground truth labels. Shape (Batch_Size, Num_Classes) (One-Hot Float).

        Returns:
            torch.Tensor: The computed scalar loss.
        """
        # Check if Deep Supervision is active (output is a tuple of multi-scale logits)
        if isinstance(outputs, (tuple, list)):
            # Unpack logits from different pyramid levels
            # p3: stride 8 (finest, main output)
            # p4: stride 16 (auxiliary)
            # p5: stride 32 (auxiliary)
            logits_p3, logits_p4, logits_p5 = outputs

            # Calculate loss for each head
            loss_p3 = self.loss_fn(logits_p3, targets)
            loss_p4 = self.loss_fn(logits_p4, targets)
            loss_p5 = self.loss_fn(logits_p5, targets)

            # Combine losses
            # We weight the auxiliary losses to ensure the main head dominates optimization
            total_loss = loss_p3 + self.aux_weight * (loss_p4 + loss_p5)
            return total_loss

        else:
            # Validation or Inference mode
            # The model returns only the main prediction (logits_p3)
            return self.loss_fn(outputs, targets)
