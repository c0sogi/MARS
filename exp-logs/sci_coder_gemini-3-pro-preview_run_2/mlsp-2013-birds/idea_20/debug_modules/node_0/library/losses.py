import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseConsistencyLoss(nn.Module):
    """
    Custom Loss Function for Siamese Temporal Consistency Regularization.

    Computes a composite loss consisting of:
    1. Weighted BCE Loss on the original input.
    2. Weighted BCE Loss on the time-shifted (rolled) input.
    3. MSE Consistency Loss between the probabilities of the two views.
    """

    def __init__(self, pos_weights=None, consistency_lambda=1.0):
        """
        Args:
            pos_weights (torch.Tensor, optional): Weights for positive examples for each class
                                                  to handle class imbalance.
            consistency_lambda (float): Weighting factor for the MSE consistency term.
        """
        super(SiameseConsistencyLoss, self).__init__()
        self.consistency_lambda = consistency_lambda

        # Initialize the base classification loss
        # BCEWithLogitsLoss combines a Sigmoid layer and the BCELoss in one single class.
        # It is more numerically stable than using a plain Sigmoid followed by a BCELoss.
        if pos_weights is not None:
            self.bce_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        else:
            self.bce_criterion = nn.BCEWithLogitsLoss()

    def forward(self, logits, logits_roll, targets):
        """
        Forward pass to compute the total loss.

        Args:
            logits (torch.Tensor): Raw output (logits) from the model for the original input.
                                   Shape: (Batch, Num_Classes)
            logits_roll (torch.Tensor): Raw output (logits) from the model for the rolled input.
                                        Shape: (Batch, Num_Classes)
            targets (torch.Tensor): Ground truth multi-hot labels.
                                    Shape: (Batch, Num_Classes)

        Returns:
            torch.Tensor: The scalar total loss.
        """
        # 1. Classification Loss for Original Input
        loss_bce_orig = self.bce_criterion(logits, targets)

        # 2. Classification Loss for Rolled Input
        # The label set is invariant to time-shifting, so we use the same targets.
        loss_bce_roll = self.bce_criterion(logits_roll, targets)

        # 3. Consistency Loss
        # We want the model's predictions (probabilities) to be similar regardless of the time shift.
        # We use MSE between the sigmoid probabilities of both views.
        probs_orig = torch.sigmoid(logits)
        probs_roll = torch.sigmoid(logits_roll)

        loss_consistency = F.mse_loss(probs_orig, probs_roll)

        # 4. Composite Loss
        total_loss = (
            loss_bce_orig + loss_bce_roll + (self.consistency_lambda * loss_consistency)
        )

        return total_loss
