import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class UnclampedTMSE(nn.Module):
    """
    Unclamped Truncated Mean Squared Error (T-MSE) for probability-space smoothing.
    Operates on Softmax probabilities (not Log-Softmax) and does not clamp the loss value,
    following the specific instructions for FISG-CN.
    """

    def __init__(self):
        super(UnclampedTMSE, self).__init__()

    def forward(self, logits, mask):
        """
        Args:
            logits: (B, T, C) Raw output logits.
            mask: (B, T) Binary mask indicating valid frames.
        Returns:
            scalar: The computed smoothness loss.
        """
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)

        # Calculate temporal difference: P_t - P_{t-1}
        # Shape: (B, T-1, C)
        diff = probs[:, 1:, :] - probs[:, :-1, :]

        # Squared Error
        mse = diff.pow(2).mean(dim=-1)  # Mean over classes -> (B, T-1)

        # Align mask
        # A transition is valid only if both t and t-1 are valid
        mask_t = mask[:, 1:] * mask[:, :-1]

        # Compute masked mean
        # Add epsilon to denominator to prevent division by zero
        loss = (mse * mask_t).sum() / (mask_t.sum() + 1e-8)

        return loss


class BoundaryLoss(nn.Module):
    """
    Binary Cross-Entropy Loss for boundary detection.
    """

    def __init__(self):
        super(BoundaryLoss, self).__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets, mask):
        """
        Args:
            logits: (B, T, 1) Boundary logits.
            targets: (B, T) Boundary targets (0 or 1).
            mask: (B, T) Binary mask.
        Returns:
            scalar: The computed boundary loss.
        """
        # Squeeze logits to (B, T) to match targets
        loss = self.loss_fn(logits.squeeze(-1), targets.float())

        # Apply mask
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)

        return loss


class FISGCNLoss(nn.Module):
    """
    Composite Loss function for Feature-Injected Supervised Gated-Cascaded Network.
    Aggregates Classification, Boundary, and Smoothness losses across multiple stages.
    """

    def __init__(self):
        super(FISGCNLoss, self).__init__()

        # Load weights from Config
        self.class_weights = Config.get_class_weights_tensor()
        self.w_cls = Config.WEIGHT_CLS
        self.w_bnd = Config.WEIGHT_BND
        self.w_smooth = Config.WEIGHT_SMOOTH
        self.bnd_window = Config.BOUNDARY_SMOOTHING_WINDOW

        # Initialize component losses
        self.ce_loss = nn.CrossEntropyLoss(weight=self.class_weights, reduction="none")
        self.bnd_loss_fn = BoundaryLoss()
        self.tmse_loss_fn = UnclampedTMSE()

    def generate_boundary_targets(self, class_targets):
        """
        Generates binary boundary targets from class labels.
        Marks transition frames as 1, others as 0.
        Applies dilation based on BOUNDARY_SMOOTHING_WINDOW.

        Args:
            class_targets: (B, T) Integer class labels.
        Returns:
            (B, T) Float tensor of boundary targets.
        """
        device = class_targets.device
        B, T = class_targets.shape

        # Identify transitions: where label at t != label at t-1
        # Pad the first frame to avoid size mismatch (assume no transition at t=0)
        # diff[b, t] is True if target[b, t] != target[b, t-1]

        # (B, T-1)
        diff = class_targets[:, 1:] != class_targets[:, :-1]

        # Create base signal (B, T)
        # We mark the frame `t` as a boundary if it differs from `t-1`
        change_signal = torch.zeros((B, T), device=device, dtype=torch.float)
        change_signal[:, 1:] = diff.float()

        # Apply dilation if window > 0
        if self.bnd_window > 0:
            # Prepare for conv1d: (B, 1, T)
            sig = change_signal.unsqueeze(1)

            # Kernel size for dilation: +/- window
            k_size = 2 * self.bnd_window + 1
            padding = self.bnd_window
            kernel = torch.ones((1, 1, k_size), device=device)

            # Convolve to dilate
            dilated = F.conv1d(sig, kernel, padding=padding)

            # Binarize
            bnd_targets = (dilated > 0.5).float().squeeze(1)
        else:
            bnd_targets = change_signal

        return bnd_targets

    def forward(self, stage_outputs, targets, mask):
        """
        Args:
            stage_outputs: List of dicts, one per stage.
                           Each dict must contain 'cls' (B, T, C) and 'bnd' (B, T, 1).
            targets: (B, T) Ground truth class indices.
            mask: (B, T) Valid frame mask.

        Returns:
            total_loss: Scalar tensor.
            metrics: Dictionary of individual loss components for logging.
        """
        total_loss = 0.0
        metrics = {}

        # Generate boundary targets once for the batch
        bnd_targets = self.generate_boundary_targets(targets)

        # Iterate over all stages (Deep Supervision)
        for i, output in enumerate(stage_outputs):
            stage_name = f"stage{i+1}"

            cls_logits = output["cls"]
            bnd_logits = output["bnd"]

            # 1. Classification Loss (Weighted Cross Entropy)
            # Flatten inputs for CrossEntropyLoss
            ce = self.ce_loss(
                cls_logits.reshape(-1, Config.NUM_CLASSES), targets.reshape(-1)
            )
            ce = ce.view(targets.shape)
            loss_cls = (ce * mask).sum() / (mask.sum() + 1e-8)

            # 2. Boundary Loss
            loss_bnd = self.bnd_loss_fn(bnd_logits, bnd_targets, mask)

            # 3. Smoothness Loss (Unclamped T-MSE)
            loss_smooth = self.tmse_loss_fn(cls_logits, mask)

            # Weighted Sum for this stage
            stage_loss = (
                (self.w_cls * loss_cls)
                + (self.w_bnd * loss_bnd)
                + (self.w_smooth * loss_smooth)
            )

            total_loss += stage_loss

            # Record metrics
            metrics[f"{stage_name}_loss"] = stage_loss.item()
            metrics[f"{stage_name}_cls"] = loss_cls.item()
            metrics[f"{stage_name}_bnd"] = loss_bnd.item()
            metrics[f"{stage_name}_smooth"] = loss_smooth.item()

        return total_loss, metrics
