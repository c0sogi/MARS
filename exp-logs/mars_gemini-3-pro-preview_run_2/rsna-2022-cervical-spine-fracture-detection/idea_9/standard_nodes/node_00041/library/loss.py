import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config
from library.model import WeightedMultilabelLoss, DiceLoss


class TriLevelFractureLoss(nn.Module):
    """
    Computes the composite loss for the Cervical Spine Fracture Detection model.

    Components:
    1. Study Loss: Weighted Multi-label Log Loss for exam-level predictions (C1-C7, Overall).
    2. Slice Loss: Binary Cross Entropy for detecting fracture presence on specific slices.
    3. Spatial Loss: Dice Loss for localizing fractures within a slice (supervised by bounding boxes).
    """

    def __init__(
        self,
        lambda_study: float = 1.0,
        lambda_slice: float = Config.LAMBDA_SLICE,
        lambda_spatial: float = Config.LAMBDA_SPATIAL,
    ):
        """
        Args:
            lambda_study (float): Weight for the study-level classification loss.
            lambda_slice (float): Weight for the slice-level auxiliary loss.
            lambda_spatial (float): Weight for the spatial attention supervision loss.
        """
        super().__init__()
        self.lambda_study = lambda_study
        self.lambda_slice = lambda_slice
        self.lambda_spatial = lambda_spatial

        # 1. Study-Level Loss (Weighted Log Loss)
        # Expects probabilities (sigmoid applied)
        self.study_criterion = WeightedMultilabelLoss()

        # 2. Slice-Level Loss (BCE)
        # Expects logits
        self.slice_criterion = nn.BCEWithLogitsLoss()

        # 3. Spatial Loss (Dice)
        # Expects probabilities (sigmoid applied)
        self.spatial_criterion = DiceLoss()

    def forward(self, outputs: dict, targets: dict) -> torch.Tensor:
        """
        Calculates the weighted sum of losses.

        Args:
            outputs (dict): Dictionary containing model outputs:
                - "study_logits": (B, 8)
                - "slice_logits": (B, Seq)
                - "spatial_logits": (B, Seq, 1, H_feat, W_feat)
            targets (dict): Dictionary containing ground truth:
                - "study_labels": (B, 8)
                - "slice_labels": (B, Seq)
                - "spatial_masks": (B, Seq, 1, H_img, W_img)

        Returns:
            torch.Tensor: The scalar total loss.
        """
        # Unpack outputs
        study_logits = outputs["study_logits"]
        slice_logits = outputs["slice_logits"]
        spatial_logits = outputs["spatial_logits"]

        # Unpack targets
        study_labels = targets["study_labels"]
        slice_labels = targets["slice_labels"]
        spatial_masks = targets["spatial_masks"]

        # --- 1. Study Loss ---
        # Apply sigmoid because WeightedMultilabelLoss expects probabilities
        pred_study = torch.sigmoid(study_logits)
        loss_study = self.study_criterion(pred_study, study_labels)

        # --- 2. Slice Loss ---
        # BCEWithLogitsLoss takes logits directly
        loss_slice = self.slice_criterion(slice_logits, slice_labels)

        # --- 3. Spatial Loss ---
        # Spatial loss is only computed for slices that have a fracture (slice_label == 1).
        # We also need to handle resolution mismatch between original masks and feature maps.

        # Dimensions
        b, s, c, h_feat, w_feat = spatial_logits.shape

        # Flatten batch and sequence dims for processing
        spatial_logits_flat = spatial_logits.view(b * s, c, h_feat, w_feat)
        spatial_masks_flat = spatial_masks.view(
            b * s, 1, spatial_masks.shape[3], spatial_masks.shape[4]
        )
        slice_labels_flat = slice_labels.view(b * s)

        # Identify positive slices (where fracture exists)
        # We use a threshold of 0.5 to determine positive labels
        pos_indices = slice_labels_flat > 0.5

        if pos_indices.sum() > 0:
            # Select only positive samples
            pred_subset = spatial_logits_flat[pos_indices]
            mask_subset = spatial_masks_flat[pos_indices]

            # Downsample the high-res mask to match the feature map size (e.g., 24x24)
            # Using nearest neighbor as masks are binary
            mask_subset_small = F.interpolate(
                mask_subset, size=(h_feat, w_feat), mode="nearest"
            )

            # Apply sigmoid to predictions
            pred_subset_sig = torch.sigmoid(pred_subset)

            # Calculate Dice Loss
            loss_spatial = self.spatial_criterion(pred_subset_sig, mask_subset_small)
        else:
            # If no fractures in the batch, spatial loss is 0
            loss_spatial = torch.tensor(0.0, device=study_logits.device)

        # --- Total Loss ---
        total_loss = (
            (self.lambda_study * loss_study)
            + (self.lambda_slice * loss_slice)
            + (self.lambda_spatial * loss_spatial)
        )

        return total_loss
