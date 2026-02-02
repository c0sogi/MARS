import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config
from library.utils import make_pad_mask


class TMSELoss(nn.Module):
    """
    Truncated Mean Squared Error (T-MSE) Loss.

    Applied to the Softmax probabilities of the model outputs to enforce
    temporal smoothness. The error is calculated between consecutive frames
    and truncated at a specified threshold to prevent gradient instability
    while penalizing jitter.
    """

    def __init__(self, threshold=4.0):
        """
        Args:
            threshold (float): The maximum allowed squared error value.
                               Errors exceeding this are clamped.
        """
        super(TMSELoss, self).__init__()
        self.threshold = threshold

    def forward(self, logits, mask):
        """
        Args:
            logits (torch.Tensor): Logits of shape (Batch, Classes, Time).
            mask (torch.Tensor): Boolean mask of shape (Batch, Time) where True indicates valid frames.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Convert logits to probabilities (Softmax, not Log-Softmax)
        probs = F.softmax(logits, dim=1)

        # Calculate temporal difference: P_t - P_{t-1}
        # Shape: (Batch, Classes, Time-1)
        # We slice time dim: 1: vs :-1
        diff = probs[:, :, 1:] - probs[:, :, :-1]

        # Squared Error
        mse = diff.pow(2)

        # Truncate (clamp) the error
        # Note: With probabilities in [0, 1], max MSE is 1.0.
        # If threshold > 1.0 (e.g. 4.0), this acts as standard MSE.
        tmse = torch.clamp(mse, max=self.threshold)

        # Apply mask
        # We need to mask the differences. diff at index t corresponds to change between t and t+1
        # (or t-1 and t depending on indexing).
        # Here diff[:,:,i] is probs[:,:,i+1] - probs[:,:,i].
        # We should consider a transition valid if BOTH frames are valid.
        # mask[:, 1:] corresponds to the validity of the "current" frame in the diff (t).
        # mask[:, :-1] corresponds to the "previous" frame.
        # Generally, if the sequence length is L, we have L-1 transitions.
        # We use mask[:, 1:] to select valid transitions ending at valid positions.

        valid_transitions = mask[:, 1:].unsqueeze(1)  # (Batch, 1, Time-1)

        # Zero out invalid transitions
        masked_tmse = tmse * valid_transitions.float()

        # Compute mean over valid transitions
        # Sum over all dims, divide by number of valid elements * classes
        num_valid = valid_transitions.sum()

        if num_valid == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        loss = masked_tmse.sum() / (num_valid * probs.size(1))

        return loss


class CombinedLoss(nn.Module):
    """
    Combined Loss Module for GMD-CRCN.

    Computes the weighted sum of losses from all three stages:
    1. Stage 1 (Generation): Weighted Cross Entropy
    2. Stage 2 (Coarse Refinement): Weighted Cross Entropy + TMSE
    3. Stage 3 (Fine Sharpening): Weighted Cross Entropy + TMSE
    """

    def __init__(self, device):
        super(CombinedLoss, self).__init__()
        self.device = device

        # Load configuration
        self.class_weights = Config.get_class_weights_tensor(device)
        self.lambda_gen = Config.LOSS_LAMBDA_GEN
        self.lambda_ref1 = Config.LOSS_LAMBDA_REF1
        self.lambda_ref2 = Config.LOSS_LAMBDA_REF2
        self.tmse_weight = Config.TMSE_WEIGHT

        # Initialize Loss Functions
        # reduction='none' allows us to apply the mask manually
        self.ce_loss = nn.CrossEntropyLoss(weight=self.class_weights, reduction="none")
        self.tmse_loss = TMSELoss(threshold=Config.TMSE_THRESHOLD)

    def compute_masked_ce(self, logits, targets, mask):
        """
        Computes masked Cross Entropy Loss.

        Args:
            logits: (Batch, Classes, Time)
            targets: (Batch, Time)
            mask: (Batch, Time)
        """
        B, C, T = logits.shape

        # Flatten tensors for CrossEntropyLoss
        # Permute logits to (Batch, Time, Classes) then reshape to (Batch*Time, Classes)
        logits_flat = logits.permute(0, 2, 1).reshape(-1, C)
        targets_flat = targets.reshape(-1)
        mask_flat = mask.reshape(-1)

        # Compute element-wise loss
        loss = self.ce_loss(logits_flat, targets_flat)

        # Apply mask
        masked_loss = loss * mask_flat.float()

        # Normalize by number of valid tokens
        num_valid = mask_flat.sum()
        if num_valid == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        return masked_loss.sum() / num_valid

    def forward(self, predictions, targets, lengths):
        """
        Forward pass for loss calculation.

        Args:
            predictions (dict): Dictionary containing model outputs:
                                - 'stage1': Logits (Batch, Classes, Time)
                                - 'stage2': Logits (Batch, Classes, Time)
                                - 'stage3': Logits (Batch, Classes, Time)
            targets (torch.Tensor): Ground truth labels (Batch, Time).
            lengths (torch.Tensor): Sequence lengths (Batch).

        Returns:
            tuple: (total_loss, loss_dict)
        """
        # Create mask for valid sequence positions
        # targets size (Batch, Time) determines max_len
        mask = make_pad_mask(lengths, max_len=targets.size(1)).to(self.device)

        # --- Stage 1: Generation ---
        # Only Cross Entropy
        p0 = predictions["stage1"]
        loss_gen = self.compute_masked_ce(p0, targets, mask)

        # --- Stage 2: Coarse Refinement ---
        # Cross Entropy + TMSE
        p1 = predictions["stage2"]
        loss_ref1_ce = self.compute_masked_ce(p1, targets, mask)
        loss_ref1_tmse = self.tmse_loss(p1, mask)
        loss_ref1 = loss_ref1_ce + (self.tmse_weight * loss_ref1_tmse)

        # --- Stage 3: Fine Sharpening ---
        # Cross Entropy + TMSE
        p2 = predictions["stage3"]
        loss_ref2_ce = self.compute_masked_ce(p2, targets, mask)
        loss_ref2_tmse = self.tmse_loss(p2, mask)
        loss_ref2 = loss_ref2_ce + (self.tmse_weight * loss_ref2_tmse)

        # --- Total Loss ---
        # Weighted sum of stage losses
        total_loss = (
            self.lambda_gen * loss_gen
            + self.lambda_ref1 * loss_ref1
            + self.lambda_ref2 * loss_ref2
        )

        # Detailed metrics for logging
        loss_dict = {
            "loss_gen": loss_gen.item(),
            "loss_ref1": loss_ref1.item(),
            "loss_ref1_ce": loss_ref1_ce.item(),
            "loss_ref1_tmse": loss_ref1_tmse.item(),
            "loss_ref2": loss_ref2.item(),
            "loss_ref2_ce": loss_ref2_ce.item(),
            "loss_ref2_tmse": loss_ref2_tmse.item(),
            "loss_total": total_loss.item(),
        }

        return total_loss, loss_dict
