import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class HybridHierarchicalLoss(nn.Module):
    """
    Implements the Hybrid Hierarchical Loss for Cervical Spine Fracture Detection.

    Components:
    1. L_MIL: Weighted Multi-label Logarithmic Loss on global max-pooled predictions.
       Weights: 1.0 for C1-C7, 7.0 for patient_overall.
    2. L_Box: Slice-Level Binary Cross Entropy on instance logits, supervised by
       bounding box masks. Applied only to samples with valid box annotations.
    """

    def __init__(self):
        super().__init__()
        self.alpha = Config.ALPHA_BOX_LOSS

        # Competition metric weights: C1-C7 = 1.0, Patient_Overall = 7.0
        # Registered as buffer to handle device movement automatically
        weights = torch.tensor([1.0] * 7 + [7.0])
        self.register_buffer("mil_weights", weights)

        # Base Loss Functions
        # MIL Loss: Weighted BCE. Reduction is mean to average over batch.
        # Note: We pass pos_weight=None, but use the 'weight' argument for class balancing
        # in the forward pass if needed, or initialize here if static.
        # PyTorch BCEWithLogitsLoss 'weight' arg is for rescaling weight given to the loss of each batch element
        # OR class if broadcastable.
        self.bce_mil = nn.BCEWithLogitsLoss(weight=self.mil_weights)

        # Box Loss: Standard BCE. Reduction is none to allow masking.
        self.bce_box = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, instance_logits, targets, box_targets=None, box_mask=None):
        """
        Args:
            instance_logits (torch.Tensor): Shape (B, Seq_Len, 7). Raw logits for C1-C7 per slice.
            targets (torch.Tensor): Shape (B, 8). Global ground truth [C1...C7, patient_overall].
            box_targets (torch.Tensor, optional): Shape (B, Seq_Len, 7). Sparse binary masks for fractures.
            box_mask (torch.Tensor, optional): Shape (B,). Binary mask indicating which samples have
                                               valid bounding box annotations (1.0 = has boxes, 0.0 = no boxes).

        Returns:
            tuple: (total_loss, metrics_dict)
        """
        # --- 1. MIL Component (Global Level) ---

        # Global Max Pooling: Aggregate slice logits to study logits
        # Shape: (B, 7)
        pooled_logits, _ = torch.max(instance_logits, dim=1)

        # Derive patient_overall logit
        # Logic: If any vertebra is fractured, patient is fractured.
        # In probability space: p_overall = max(p_c1, ... p_c7)
        # In logit space: logit_overall = max(logit_c1, ... logit_c7)
        # Shape: (B, 1)
        patient_logit, _ = torch.max(pooled_logits, dim=1, keepdim=True)

        # Concatenate to form full prediction vector: [C1...C7, patient_overall]
        # Shape: (B, 8)
        global_logits = torch.cat([pooled_logits, patient_logit], dim=1)

        # Calculate Weighted MIL Loss
        loss_mil = self.bce_mil(global_logits, targets.float())

        # --- 2. Box Component (Slice Level) ---

        loss_box = torch.tensor(0.0, device=instance_logits.device)

        # Only compute box loss if targets and mask are provided
        if box_targets is not None and box_mask is not None:
            # Flatten sequence dims for BCE if needed, or keep as is.
            # instance_logits: (B, S, 7), box_targets: (B, S, 7)

            # Compute raw BCE loss per element
            # Shape: (B, S, 7)
            raw_box_loss = self.bce_box(instance_logits, box_targets.float())

            # Aggregate loss per sample (average over sequence and classes)
            # Shape: (B,)
            per_sample_box_loss = raw_box_loss.mean(dim=(1, 2))

            # Apply mask: Zero out loss for samples without box annotations
            # box_mask shape: (B,)
            masked_loss = per_sample_box_loss * box_mask

            # Normalize by the number of valid samples to avoid gradient dilution
            num_valid_samples = box_mask.sum()

            if num_valid_samples > 0:
                loss_box = masked_loss.sum() / num_valid_samples
            else:
                loss_box = torch.tensor(0.0, device=instance_logits.device)

        # --- 3. Total Loss ---
        total_loss = loss_mil + self.alpha * loss_box

        metrics = {
            "loss": total_loss.item(),
            "loss_mil": loss_mil.item(),
            "loss_box": loss_box.item(),
        }

        return total_loss, metrics
