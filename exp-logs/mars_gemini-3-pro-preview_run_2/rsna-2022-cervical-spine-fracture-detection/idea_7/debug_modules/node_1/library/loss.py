import torch
import torch.nn as nn
from library.config import Config


class HybridLoss(nn.Module):
    """
    Computes the hybrid loss for the 2.5D Multi-Scale Fracture Detection model.

    The loss consists of two components:
    1. Study Loss: Weighted Multi-Label Logarithmic Loss for the 8 study-level targets.
       Includes positive class weighting to handle imbalance.
    2. Slice Loss: Binary Cross Entropy for the auxiliary dense slice predictions.
       This is masked to only apply to studies with valid bounding box annotations.
    """

    def __init__(self):
        super(HybridLoss, self).__init__()

        # --- Configuration ---
        self.lambda_slice = Config.LAMBDA_SLICE_LOSS

        # --- Study Loss Setup ---
        # Weights for the 8 classes (C1-C7, Overall)
        # We register as buffer to ensure it moves to device with the module
        weights = Config.LOSS_WEIGHTS
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights)
        self.register_buffer("class_weights", weights)

        # Positive class weight (scalar) for sensitivity
        pos_weight = torch.tensor([Config.POS_WEIGHT_STUDY])
        self.register_buffer("pos_weight", pos_weight)

        # Initialize Study Criterion
        # weight argument handles the column-specific weighting (importance of 'overall')
        # pos_weight argument handles the imbalance (sensitivity)
        self.study_criterion = nn.BCEWithLogitsLoss(
            weight=self.class_weights, pos_weight=self.pos_weight, reduction="mean"
        )

        # --- Slice Loss Setup ---
        # Initialize Slice Criterion
        # reduction='none' is required to apply the mask manually
        self.slice_criterion = nn.BCEWithLogitsLoss(reduction="none")

    def forward(
        self, study_logits, study_targets, slice_logits, slice_targets, slice_mask
    ):
        """
        Calculates the total hybrid loss.

        Args:
            study_logits (torch.Tensor): Predicted logits for study-level classes.
                                         Shape: (Batch, 8)
            study_targets (torch.Tensor): Ground truth labels for study-level classes.
                                          Shape: (Batch, 8)
            slice_logits (torch.Tensor): Predicted logits for slice-level auxiliary head.
                                         Shape: (Batch, Seq_Len) or (Batch, Seq_Len, 1)
            slice_targets (torch.Tensor): Ground truth labels for slice-level targets.
                                          Shape: (Batch, Seq_Len) or (Batch, Seq_Len, 1)
            slice_mask (torch.Tensor): Boolean/Binary mask indicating which studies have
                                       bounding box annotations.
                                       Shape: (Batch,) or (Batch, 1)

        Returns:
            torch.Tensor: The scalar total loss.
        """
        # --- 1. Study Level Loss ---
        # Ensure targets are float for BCE
        study_targets = study_targets.float()

        # Calculate weighted BCE for study targets
        loss_study = self.study_criterion(study_logits, study_targets)

        # --- 2. Slice Level Loss (Auxiliary) ---
        # Ensure shapes match for broadcasting/calculation
        if slice_logits.dim() == 3:
            slice_logits = slice_logits.squeeze(-1)  # (B, Seq, 1) -> (B, Seq)
        if slice_targets.dim() == 3:
            slice_targets = slice_targets.squeeze(-1)  # (B, Seq, 1) -> (B, Seq)

        slice_targets = slice_targets.float()

        # Compute raw BCE loss per element (Batch, Seq_Len)
        raw_slice_loss = self.slice_criterion(slice_logits, slice_targets)

        # Prepare mask
        # slice_mask indicates if a study (row in batch) has valid bbox data.
        # Shape (B,) -> Expand to (B, Seq_Len) to mask the entire sequence for that study.
        if slice_mask.dim() == 1:
            slice_mask = slice_mask.unsqueeze(1)  # (B, 1)

        # Expand mask to match sequence length
        # mask shape becomes (B, Seq_Len)
        expanded_mask = slice_mask.expand_as(raw_slice_loss)

        # Apply mask: Zero out loss for studies without bounding boxes
        masked_slice_loss = raw_slice_loss * expanded_mask

        # Average the loss over valid elements only
        # Add epsilon to avoid division by zero if no studies in batch have bboxes
        valid_elements = expanded_mask.sum()
        if valid_elements > 0:
            loss_slice = masked_slice_loss.sum() / valid_elements
        else:
            loss_slice = torch.tensor(0.0, device=study_logits.device)

        # --- 3. Total Loss ---
        total_loss = loss_study + (self.lambda_slice * loss_slice)

        return total_loss
