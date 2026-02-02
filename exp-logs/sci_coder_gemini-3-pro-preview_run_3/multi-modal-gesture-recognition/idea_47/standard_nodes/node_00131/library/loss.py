import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class LogSpaceSmoothingLoss(nn.Module):
    """
    Implements Truncated MSE on log-probabilities to enforce temporal smoothness.

    Formula:
        L = mean( clamp( (log(P_t) - log(P_{t-1}))^2, 0, threshold ) )

    This penalizes rapid fluctuations in prediction confidence between adjacent frames,
    acting as a regularizer against jitter/over-segmentation. The truncation (clamping)
    ensures that genuine sharp transitions (gesture boundaries) are not penalized excessively.
    """

    def __init__(self, threshold=1.0):
        super(LogSpaceSmoothingLoss, self).__init__()
        self.threshold = threshold

    def forward(self, logits):
        """
        Args:
            logits (torch.Tensor): Raw output logits of shape (Batch, Time, NumClasses).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Convert logits to log-probabilities
        log_probs = F.log_softmax(logits, dim=2)

        # Calculate temporal differences: log(P_t) - log(P_{t-1})
        # Slice to align t and t-1
        # Shape: (Batch, Time-1, NumClasses)
        diff = log_probs[:, 1:, :] - log_probs[:, :-1, :]

        # Squared Error
        mse = diff.pow(2)

        # Truncate the error to the threshold
        truncated_mse = torch.clamp(mse, max=self.threshold)

        # Return the mean over all dimensions (Batch, Time, Classes)
        return truncated_mse.mean()


class CascadedLoss(nn.Module):
    """
    Aggregates losses from the three stages of the DGC-KN model (Deep Supervision).

    Structure:
        L_total = Sum( w_i * (L_CE_i + lambda * L_Smooth_i) ) for i in [1, 2, 3]

    Components:
        - Weighted Cross Entropy: Handles class imbalance (background vs gestures).
        - Log-Space Smoothing: Regularizes temporal consistency.
    """

    def __init__(self):
        super(CascadedLoss, self).__init__()

        # Load configuration
        self.stage_weights = Config.LOSS_STAGE_WEIGHTS
        self.smoothing_lambda = Config.SMOOTHING_LAMBDA

        # Initialize Weighted Cross Entropy
        # Weights are moved to the configured device automatically by the Config helper
        class_weights = Config.get_class_weights()
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)

        # Initialize Smoothing Loss
        self.smooth_loss = LogSpaceSmoothingLoss(threshold=Config.SMOOTHING_THRESHOLD)

    def forward(self, model_outputs, targets):
        """
        Args:
            model_outputs (dict): Dictionary containing 'logits_1', 'logits_2', 'logits_3'.
            targets (torch.Tensor): Ground truth labels of shape (Batch, Time).

        Returns:
            tuple: (total_loss, metrics_dict)
        """
        total_loss = 0.0
        metrics = {}

        # Flatten targets for CrossEntropy: (Batch * Time)
        # Note: We assume targets are dense.
        targets_flat = targets.view(-1)

        # Keys corresponding to the model output dictionary
        stage_keys = ["logits_1", "logits_2", "logits_3"]

        for i, key in enumerate(stage_keys):
            if key not in model_outputs:
                continue

            logits = model_outputs[key]  # (Batch, Time, NumClasses)

            # --- 1. Cross Entropy Loss ---
            # Reshape logits to (Batch * Time, NumClasses)
            B, T, C = logits.shape
            logits_flat = logits.reshape(-1, C)

            # Ensure targets match the flattened length (safety check for edge cases)
            # In standard training, they should match exactly.
            current_ce_loss = self.ce_loss(logits_flat, targets_flat)

            # --- 2. Smoothing Loss ---
            # Calculated on the 3D logits to preserve temporal structure
            current_smooth_loss = self.smooth_loss(logits)

            # --- 3. Stage Aggregation ---
            stage_loss = current_ce_loss + (self.smoothing_lambda * current_smooth_loss)

            # Apply stage weight (Deep Supervision)
            weight = self.stage_weights[i] if i < len(self.stage_weights) else 1.0
            total_loss += weight * stage_loss

            # --- 4. Metrics Logging ---
            metrics[f"loss_stage_{i+1}"] = stage_loss.item()
            metrics[f"ce_stage_{i+1}"] = current_ce_loss.item()
            metrics[f"smooth_stage_{i+1}"] = current_smooth_loss.item()

        return total_loss, metrics
