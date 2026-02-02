import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class JointAlignedLoss(nn.Module):
    """
    Implements the Joint Aligned Loss for the Multi-Task RNA Degradation Model.

    This loss function combines:
    1. Masked Mean Squared Error (MSE) for the regression task.
       - Applied strictly to the first 68 positions (seq_scored).
       - Applied strictly to the 3 scored columns: reactivity, deg_Mg_pH10, deg_Mg_50C.
    2. Cross Entropy Loss for the reconstruction task (MLM).
       - Applied only to masked sequence positions.
       - Acts as a structural regularizer.

    Equation: L = MSE_reg + lambda * CE_recon
    """

    def __init__(self, config=Config()):
        super().__init__()
        self.config = config
        # Reconstruction loss ignores tokens labeled with -100 (unmasked positions)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, reg_pred, recon_pred, targets, mask_labels):
        """
        Calculates the joint loss.

        Args:
            reg_pred (torch.Tensor): Model regression predictions of shape (B, Seq_Len, 3).
            recon_pred (torch.Tensor): Model reconstruction logits of shape (B, Seq_Len, 4).
            targets (torch.Tensor): Ground truth targets of shape (B, Scored_Len, 5).
                                    Contains all 5 columns.
            mask_labels (torch.Tensor): Ground truth labels for masked tokens of shape (B, Seq_Len).
                                        Unmasked positions are -100.

        Returns:
            tuple: (total_loss, mse_loss, ce_loss)
        """
        # ==============================================================================
        # 1. Regression Loss (Scored Columns Only)
        # ==============================================================================

        # The model outputs predictions for the full sequence length (e.g., 107),
        # but we only score the first 68 positions.
        # Slice predictions: (B, 107, 3) -> (B, 68, 3)
        scored_preds = reg_pred[:, : self.config.SCORED_LEN, :]

        # The dataset provides 5 target columns, but we only train on the 3 scored ones.
        # The indices are defined in config (0: reactivity, 1: deg_Mg_pH10, 3: deg_Mg_50C).
        # Slice targets: (B, 68, 5) -> (B, 68, 3)
        scored_targets = targets[
            :, : self.config.SCORED_LEN, self.config.SCORED_INDICES
        ]

        # Calculate Mean Squared Error
        mse_loss = F.mse_loss(scored_preds, scored_targets)

        # ==============================================================================
        # 2. Reconstruction Loss (Masked Language Modeling)
        # ==============================================================================

        # CrossEntropyLoss expects input (B, C, L) and target (B, L).
        # Transpose logits: (B, L, 4) -> (B, 4, L)
        recon_logits = recon_pred.transpose(1, 2)

        # Calculate Cross Entropy (ignores -100 in mask_labels)
        ce_loss = self.ce_loss(recon_logits, mask_labels)

        # ==============================================================================
        # 3. Total Loss
        # ==============================================================================

        total_loss = mse_loss + self.config.LAMBDA_RECON * ce_loss

        return total_loss, mse_loss, ce_loss
