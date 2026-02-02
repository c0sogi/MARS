import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class HybridLoss(nn.Module):
    """
    Calibrated Dual-Objective Loss Function.
    Combines Weighted Multi-Label Logarithmic Loss for classification
    and Supervised Attention Guidance (KL Divergence) for localization.
    """

    def __init__(self):
        super().__init__()
        # Weights according to competition metric:
        # C1-C7: 1/7 (~0.142), Patient Overall: 1.0
        # The metric averages all rows, so we apply these weights per column.
        # We initialize on CPU; they will be moved to the correct device in forward().
        self.class_weights = torch.tensor([1.0 / 7.0] * 7 + [1.0], dtype=torch.float32)
        self.attn_lambda = Config.ATTENTION_LAMBDA

    def forward(self, outputs, targets, attn_targets, has_bbox):
        """
        Args:
            outputs (dict): Model outputs containing:
                - 'logits': (Batch, 8)
                - 'attn_weights': (Batch, 8, Seq_Len)
            targets (torch.Tensor): Classification targets (Batch, 8).
            attn_targets (torch.Tensor): Attention masks (Batch, 7, Seq_Len).
            has_bbox (torch.Tensor): Flag indicating bbox presence (Batch, 1).

        Returns:
            tuple: (total_loss, classification_loss, attention_loss)
        """
        logits = outputs["logits"]
        pred_attn = outputs["attn_weights"]

        # Ensure class weights are on the same device as the model outputs
        if self.class_weights.device != logits.device:
            self.class_weights = self.class_weights.to(logits.device)

        # ---------------------------------------------------------------------
        # 1. Classification Loss (L_study)
        # ---------------------------------------------------------------------
        # We use BCEWithLogitsLoss with reduction='none' to apply custom weights.
        # Note: pos_weight is NOT used. Training on the natural class balance
        # ensures the predicted probabilities are well-calibrated, which minimizes Log Loss.
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # Apply competition-specific weights (Overall=1.0, C-levels=1/7)
        weighted_bce = bce_loss * self.class_weights

        # Average over all entries (batch * classes) to match the global average
        # nature of the competition metric (which averages over all rows).
        loss_cls = weighted_bce.mean()

        # ---------------------------------------------------------------------
        # 2. Attention Supervision Loss (L_attn)
        # ---------------------------------------------------------------------
        loss_attn = torch.tensor(0.0, device=logits.device)

        # Only compute attention loss if there are bounding boxes in the batch
        if has_bbox.sum() > 0:
            # We only supervise C1-C7 (indices 0-6).
            # Index 7 is patient_overall which has no explicit bbox mask in dataset.
            pred_attn_subset = pred_attn[:, :7, :]  # Shape: (B, 7, S)

            # Masking Logic:
            # We only supervise heads where:
            # 1. The study has bounding box annotations (has_bbox == 1)
            # 2. The specific vertebrae is fractured (targets == 1)
            #    (The dataset logic only generates Gaussian masks for fractured levels)

            # Expand has_bbox to match channel dimensions: (B, 1) -> (B, 7)
            study_mask = has_bbox.view(-1, 1).expand(-1, 7) > 0.5

            # Check fracture targets: (B, 7)
            fracture_mask = targets[:, :7] > 0.5

            # Combined mask: Valid supervision targets
            valid_mask = study_mask & fracture_mask

            if valid_mask.sum() > 0:
                # Select valid predictions and targets
                # Flatten batch and channel dims -> (N_valid, S)
                p_valid = pred_attn_subset[valid_mask]
                t_valid = attn_targets[valid_mask]

                # Normalize targets to be a valid probability distribution (sum=1).
                # The dataset provides max=1 Gaussians (heatmaps).
                # We normalize them to use with KL Divergence.
                t_sum = t_valid.sum(dim=1, keepdim=True)
                t_valid_dist = t_valid / (t_sum + 1e-6)

                # KL Divergence: sum(target * (log(target) - log(pred)))
                # PyTorch F.kl_div expects input as log-probabilities.
                # p_valid comes from Softmax (in model.py), so we take log.
                # Add epsilon for numerical stability.
                log_p_valid = torch.log(p_valid + 1e-9)

                # reduction='batchmean' averages over the batch dimension (N_valid)
                loss_attn = F.kl_div(log_p_valid, t_valid_dist, reduction="batchmean")

        # ---------------------------------------------------------------------
        # 3. Total Loss
        # ---------------------------------------------------------------------
        total_loss = loss_cls + self.attn_lambda * loss_attn

        return total_loss, loss_cls, loss_attn
