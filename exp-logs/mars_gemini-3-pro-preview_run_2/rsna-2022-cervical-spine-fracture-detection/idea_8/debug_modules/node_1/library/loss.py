import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class TriLevelLoss(nn.Module):
    """
    Implements the Tri-Level Loss function for the 2.5D Dual-Attention Network.

    This loss function aggregates three distinct objectives:
    1. Study Loss: Weighted Multi-Label Log Loss for the final classification,
       incorporating both competition-specific class weights and sensitivity adjustments.
    2. Slice Loss: Binary Cross Entropy for auxiliary slice-level fracture detection.
    3. Spatial Loss: Masked Dice Loss for supervising the spatial attention mechanism
       using ground-truth bounding boxes.
    """

    def __init__(self, config=Config):
        super().__init__()
        self.config = config

        # --- Loss Component Weights ---
        self.lambda_slice = config.LAMBDA_SLICE
        self.lambda_spatial = config.LAMBDA_SPATIAL

        # --- Study Loss Configuration ---
        # 1. Class Weights (Competition Metric Heuristic)
        # Typically: 1.0 for C1-C7, 7.0 for patient_overall
        num_classes = len(config.TARGET_COLS)
        class_weights = torch.ones(num_classes)

        if "patient_overall" in config.TARGET_COLS:
            try:
                idx = config.TARGET_COLS.index("patient_overall")
                class_weights[idx] = 7.0
            except ValueError:
                pass  # Should not happen given check above

        # Register as buffer to handle device movement automatically
        self.register_buffer("class_weights", class_weights)

        # 2. Positive Class Weight (Sensitivity Adjustment)
        # Used to upweight positive samples in BCE to handle class imbalance
        pos_weight_val = config.POS_WEIGHT_STUDY
        self.pos_weights = torch.tensor([pos_weight_val] * num_classes)
        self.register_buffer("pos_weights", self.pos_weights)

    def dice_loss(self, logits, targets, smooth=1e-6):
        """
        Computes the Dice Loss for spatial masks.

        Args:
            logits: Predicted logits of shape (N, 1, H, W)
            targets: Ground truth binary masks of shape (N, 1, H, W)
            smooth: Smoothing factor to avoid division by zero

        Returns:
            Scalar Dice Loss (1 - Dice Coefficient)
        """
        # Apply sigmoid to convert logits to probabilities
        probs = torch.sigmoid(logits)

        # Flatten spatial dimensions: (N, 1, H, W) -> (N, H*W)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)

        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice.mean()

    def forward(self, preds, targets):
        """
        Computes the total weighted loss.

        Args:
            preds: Dictionary containing model predictions:
                - 'study_logits': (B, Num_Classes)
                - 'slice_logits': (B, Seq_Len)
                - 'spatial_logits': (B, Seq_Len, 1, H_feat, W_feat)
            targets: Dictionary containing ground truth:
                - 'label_study': (B, Num_Classes)
                - 'label_slice': (B, Seq_Len)
                - 'label_spatial': (B, Seq_Len, 1, H_img, W_img)

        Returns:
            Total loss tensor (scalar).
        """
        device = preds["study_logits"].device

        # ---------------------------------------------------------------------
        # 1. Study Level Loss
        # ---------------------------------------------------------------------
        study_logits = preds["study_logits"]
        study_targets = targets["label_study"]

        # BCEWithLogitsLoss combines Sigmoid and BCE.
        # pos_weight handles the imbalance (sensitivity).
        # weight handles the competition metric importance (overall vs subtypes).
        study_loss = F.binary_cross_entropy_with_logits(
            study_logits,
            study_targets,
            pos_weight=self.pos_weights,
            weight=self.class_weights,
            reduction="mean",
        )

        # ---------------------------------------------------------------------
        # 2. Slice Level Loss (Auxiliary)
        # ---------------------------------------------------------------------
        slice_logits = preds["slice_logits"]
        slice_targets = targets["label_slice"]

        slice_loss = F.binary_cross_entropy_with_logits(
            slice_logits, slice_targets, reduction="mean"
        )

        # ---------------------------------------------------------------------
        # 3. Spatial Attention Loss (Auxiliary)
        # ---------------------------------------------------------------------
        spatial_logits = preds["spatial_logits"]  # (B, S, 1, H', W')
        spatial_targets = targets["label_spatial"]  # (B, S, 1, H, W)

        # Flatten Batch and Sequence dimensions to process slices independently
        b, s, c, h_prime, w_prime = spatial_logits.shape
        spatial_logits_flat = spatial_logits.view(b * s, c, h_prime, w_prime)
        spatial_targets_flat = spatial_targets.view(
            b * s, c, *spatial_targets.shape[-2:]
        )
        slice_targets_flat = targets["label_slice"].view(b * s)

        # Masked Supervision: Only calculate spatial loss for slices that actually
        # contain a fracture (where we have a valid bounding box mask).
        pos_mask = slice_targets_flat > 0.5

        if pos_mask.sum() > 0:
            # Filter for relevant slices
            rel_logits = spatial_logits_flat[pos_mask]
            rel_targets = spatial_targets_flat[pos_mask]

            # Upsample logits to match target resolution (e.g., 24x24 -> 384x384)
            # Use bilinear interpolation for feature maps
            rel_logits_up = F.interpolate(
                rel_logits,
                size=rel_targets.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            spatial_loss = self.dice_loss(rel_logits_up, rel_targets)
        else:
            # If no fractures in batch, spatial loss is zero
            spatial_loss = torch.tensor(0.0, device=device)

        # ---------------------------------------------------------------------
        # Total Loss Aggregation
        # ---------------------------------------------------------------------
        total_loss = (
            study_loss
            + (self.lambda_slice * slice_loss)
            + (self.lambda_spatial * spatial_loss)
        )

        return total_loss
