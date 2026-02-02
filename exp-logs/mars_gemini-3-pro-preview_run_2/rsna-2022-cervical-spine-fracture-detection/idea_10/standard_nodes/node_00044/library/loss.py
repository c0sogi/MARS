import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class CervicalSpineLoss(nn.Module):
    """
    Calibrated Multi-Task Loss for Cervical Spine Fracture Detection.

    This module implements the objective function for Idea 10, combining:
    1. Study Loss: Weighted Multi-Label Log Loss (Calibration focused, no aggressive pos_weight).
    2. Slice Fracture Loss: Masked BCE for fracture detection on slices (Dense supervision).
    3. Spatial Attention Loss: Masked Dice Loss for fracture localization (Spatial guidance).
    4. Anatomy Loss: Masked Cross-Entropy for vertebral level identification (Anatomical injection).
    """

    def __init__(self):
        super(CervicalSpineLoss, self).__init__()
        self.device = Config.DEVICE

        # Hyperparameters from Config
        self.lambda_fracture = Config.LAMBDA_FRACTURE
        self.lambda_spatial = Config.LAMBDA_SPATIAL
        self.lambda_anatomy = Config.LAMBDA_ANATOMY

        # 1. Study Loss Setup
        # Weights: C1-C7 = 1.0, Patient Overall = 7.0
        # This aligns with the competition metric weighting.
        self.study_weights = torch.tensor([1.0] * 7 + [7.0], device=self.device)

        # We use pos_weight=1.0 (or Config value) to prioritize probabilistic calibration.
        # reduction='mean' averages the weighted element-wise loss, matching the metric definition.
        self.study_loss_fn = nn.BCEWithLogitsLoss(
            weight=self.study_weights,
            pos_weight=torch.tensor([Config.POS_WEIGHT], device=self.device),
            reduction="mean",
        )

        # 2. Slice Fracture Loss Setup
        # reduction='none' allows us to apply masking based on supervision availability
        self.slice_loss_fn = nn.BCEWithLogitsLoss(reduction="none")

        # 3. Anatomy Loss Setup
        # reduction='none' allows us to apply masking based on segmentation availability
        self.anatomy_loss_fn = nn.CrossEntropyLoss(reduction="none")

    def forward(self, predictions, targets):
        """
        Calculates the weighted sum of all loss components.

        Args:
            predictions (dict):
                - 'study_logits': (B, 8)
                - 'slice_fracture_logits': (B, Seq, 1)
                - 'spatial_maps': (B, Seq, H, W) [Optional, for spatial loss]
                - 'anatomy_logits': (B, Seq, 8) [Optional, for anatomy loss]
            targets (dict):
                - 'study_labels': (B, 8)
                - 'slice_fracture_labels': (B, Seq)
                - 'spatial_masks': (B, Seq, H, W) [Optional]
                - 'anatomy_labels': (B, Seq) [Optional]
                - 'has_bbox': (B,) Boolean indicating if bbox supervision exists
                - 'has_segmentation': (B,) Boolean indicating if seg supervision exists

        Returns:
            loss (Tensor): Scalar total loss
            metrics (dict): Dictionary of individual loss components for logging
        """
        # --- Unpack Inputs ---
        study_logits = predictions["study_logits"]
        study_labels = targets["study_labels"].float()

        # --- 1. Study Loss ---
        # Calculates the weighted binary cross entropy for the 8 study-level targets
        loss_study = self.study_loss_fn(study_logits, study_labels)

        # --- 2. Slice Fracture Loss ---
        # We need to determine which slices are valid for supervision.
        # Strategy:
        # - If has_bbox=True: We trust slice_labels (generated from bbox).
        # - If has_bbox=False AND patient_overall=0: We trust slice_labels (all 0).
        # - If has_bbox=False AND patient_overall=1: Ambiguous location, ignore slices.

        slice_logits = predictions["slice_fracture_logits"].squeeze(-1)  # (B, Seq)
        slice_labels = targets["slice_fracture_labels"].float()  # (B, Seq)
        has_bbox = targets.get(
            "has_bbox", torch.zeros(study_logits.size(0), device=self.device).bool()
        )

        # Identify negative patients (last column is patient_overall)
        is_negative_patient = study_labels[:, -1] == 0

        # Valid mask: Either has bbox OR is a confirmed negative patient
        valid_slice_mask = (has_bbox | is_negative_patient).float()  # (B,)

        # Expand mask for sequence length: (B, Seq)
        valid_slice_mask_seq = valid_slice_mask.unsqueeze(1).expand_as(slice_labels)

        # Compute BCE
        slice_bce = self.slice_loss_fn(slice_logits, slice_labels)

        # Apply mask and average
        # Add epsilon to denominator to avoid div by zero if batch has no valid supervision
        loss_slice = (slice_bce * valid_slice_mask_seq).sum() / (
            valid_slice_mask_seq.sum() + 1e-6
        )

        # --- 3. Spatial Attention Loss (Dice) ---
        loss_spatial = torch.tensor(0.0, device=self.device)
        spatial_maps = predictions.get("spatial_maps")
        spatial_masks = targets.get("spatial_masks")

        if spatial_maps is not None and spatial_masks is not None:
            # Only supervise samples that have bounding boxes
            valid_indices = torch.nonzero(has_bbox).squeeze(1)

            if valid_indices.numel() > 0:
                # Filter valid items
                pred_maps = spatial_maps[valid_indices]  # (N_valid, Seq, H, W)
                true_masks = spatial_masks[
                    valid_indices
                ].float()  # (N_valid, Seq, H, W)

                # Apply Sigmoid to logits
                pred_probs = torch.sigmoid(pred_maps)

                # Flatten for Dice calculation
                pred_flat = pred_probs.view(pred_probs.size(0), -1)
                true_flat = true_masks.view(true_masks.size(0), -1)

                intersection = (pred_flat * true_flat).sum(dim=1)
                union = pred_flat.sum(dim=1) + true_flat.sum(dim=1)

                dice_score = (2.0 * intersection + 1e-6) / (union + 1e-6)
                loss_spatial = 1.0 - dice_score.mean()

        # --- 4. Anatomy Loss (Cross Entropy) ---
        loss_anatomy = torch.tensor(0.0, device=self.device)
        anatomy_logits = predictions.get("anatomy_logits")
        anatomy_labels = targets.get("anatomy_labels")
        has_segmentation = targets.get(
            "has_segmentation",
            torch.zeros(study_logits.size(0), device=self.device).bool(),
        )

        if anatomy_logits is not None and anatomy_labels is not None:
            # Only supervise samples that have segmentation masks
            valid_indices = torch.nonzero(has_segmentation).squeeze(1)

            if valid_indices.numel() > 0:
                valid_logits = anatomy_logits[valid_indices]  # (N_valid, Seq, 8)
                valid_labels = anatomy_labels[valid_indices].long()  # (N_valid, Seq)

                # Reshape for CrossEntropy: (N_samples, C) vs (N_samples)
                # Flatten sequence dimension
                valid_logits_flat = valid_logits.view(-1, Config.ANATOMY_CLASSES)
                valid_labels_flat = valid_labels.view(-1)

                # Compute CE
                loss_anatomy = self.anatomy_loss_fn(
                    valid_logits_flat, valid_labels_flat
                ).mean()

        # --- Total Loss ---
        total_loss = (
            loss_study
            + self.lambda_fracture * loss_slice
            + self.lambda_spatial * loss_spatial
            + self.lambda_anatomy * loss_anatomy
        )

        metrics = {
            "loss_total": total_loss.item(),
            "loss_study": loss_study.item(),
            "loss_slice": loss_slice.item(),
            "loss_spatial": loss_spatial.item(),
            "loss_anatomy": loss_anatomy.item(),
        }

        return total_loss, metrics
