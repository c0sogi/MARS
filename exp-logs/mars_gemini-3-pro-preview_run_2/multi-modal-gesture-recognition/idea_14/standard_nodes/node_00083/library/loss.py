import torch
import torch.nn as nn
import torch.nn.functional as F
from library.utils import get_device


class MaskedWeightedCrossEntropy(nn.Module):
    """
    Weighted Cross Entropy Loss that ignores padded frames and applies
    specific class weights (0.1 for Background, 1.0 for Gestures).
    """

    def __init__(self, num_classes: int = 21, background_weight: float = 0.1):
        super(MaskedWeightedCrossEntropy, self).__init__()
        self.num_classes = num_classes

        # Construct weights: Class 0 (Background) gets 0.1, others get 1.0
        weights = torch.ones(num_classes)
        weights[0] = background_weight
        self.register_buffer("weights", weights)

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, C, T) Raw scores (before Softmax).
            targets: (B, T) Integer class labels.
            mask: (B, T) Boolean or Binary mask (1 for valid, 0 for padding).
        """
        # CrossEntropyLoss expects (B, C, T) for logits and (B, T) for targets
        # reduction='none' to apply mask manually
        ce_loss = F.cross_entropy(
            logits, targets, weight=self.weights, reduction="none"
        )

        # Apply mask
        masked_loss = ce_loss * mask

        # Normalize by the number of valid frames
        # Add epsilon to avoid division by zero
        total_valid_frames = torch.sum(mask) + 1e-8

        return torch.sum(masked_loss) / total_valid_frames


class TMSELoss(nn.Module):
    """
    Truncated Mean Squared Error (T-MSE) for unconditional probability-space smoothing.
    Applied to Softmax probabilities to encourage smooth transitions.
    """

    def __init__(self, threshold: float = 0.15):
        super(TMSELoss, self).__init__()
        self.threshold = threshold
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            probs: (B, C, T) Softmax probabilities.
            mask: (B, T) Sequence mask.
        """
        # Calculate temporal gradients: P_t - P_{t-1}
        # Shape: (B, C, T-1)
        diff = probs[:, :, 1:] - probs[:, :, :-1]

        # Clamp gradients to threshold to avoid penalizing sharp, valid transitions too heavily
        # Note: We clamp the magnitude of the difference
        clamped_diff = torch.clamp(diff, min=-self.threshold, max=self.threshold)

        # Compute MSE on clamped differences
        loss = clamped_diff**2

        # Apply mask (align mask to T-1)
        # If frame t is valid, we assume the transition t-1 -> t is valid for smoothing
        # We use mask[:, 1:] corresponding to the 'current' frame in the difference
        mask_sliced = mask[:, 1:].unsqueeze(1)  # (B, 1, T-1)

        masked_loss = loss * mask_sliced

        # Normalize
        total_valid_elements = torch.sum(mask_sliced) * probs.shape[1] + 1e-8

        return torch.sum(masked_loss) / total_valid_elements


class DeepSupervisionLoss(nn.Module):
    """
    Aggregates losses from all model stages.
    Includes explicit boundary supervision (Cite solution_lesson_node_00073, solution_lesson_node_00082).
    """

    def __init__(
        self,
        num_classes: int = 21,
        smoothing_weight: float = 0.15,
        boundary_weight: float = 1.0,
    ):
        super(DeepSupervisionLoss, self).__init__()
        self.cls_loss_fn = MaskedWeightedCrossEntropy(num_classes=num_classes)
        self.smooth_loss_fn = TMSELoss(threshold=0.15)
        self.bnd_loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        self.smoothing_weight = smoothing_weight
        self.boundary_weight = boundary_weight
        self.num_classes = num_classes

    def forward(
        self,
        outputs: list,
        targets: torch.Tensor,
        boundaries: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            outputs: List of tuples (cls_logits, bnd_logits) from each stage.
            targets: (B, T) Class labels.
            boundaries: (B, T) Boundary labels (0 or 1).
            mask: (B, T)
        """
        total_loss = 0.0

        for cls_logits, bnd_logits in outputs:
            # 1. Classification Loss
            c_loss = self.cls_loss_fn(cls_logits, targets, mask)

            # 2. Boundary Loss (Explicit Supervision)
            # bnd_logits: (B, 1, T) -> squeeze to (B, T)
            b_loss = self.bnd_loss_fn(bnd_logits.squeeze(1), boundaries)
            # Masked mean
            b_loss = (b_loss * mask).sum() / (mask.sum() + 1e-8)

            # 3. Smoothing Loss (T-MSE) on Class Probabilities
            probs = F.softmax(cls_logits, dim=1)
            s_loss = self.smooth_loss_fn(probs, mask)

            # Aggregate
            total_loss += (
                c_loss
                + (self.boundary_weight * b_loss)
                + (self.smoothing_weight * s_loss)
            )

        return total_loss
