import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import calculate_jigsaw_metrics


class JigsawEvaluator:
    """
    Evaluator class to accumulate predictions and targets during validation/testing
    and compute the competition metrics.
    """

    def __init__(self):
        self.predictions = []
        self.targets = []
        self.identities = []

    def update(self, logits, targets, identities):
        """
        Accumulates batch results.

        Args:
            logits (torch.Tensor or np.ndarray): Model outputs (logits).
            targets (torch.Tensor or np.ndarray): Ground truth toxicity scores.
            identities (torch.Tensor or np.ndarray): Identity attribute scores.
        """
        # Convert to numpy and detach if necessary
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        if isinstance(identities, torch.Tensor):
            identities = identities.detach().cpu().numpy()

        # Apply sigmoid to logits to get probabilities in [0, 1] range
        # This converts the raw model output into the fractional probability expected
        # for analysis and by the metric calculation (though AUC is rank-invariant).
        preds = 1.0 / (1.0 + np.exp(-logits))

        self.predictions.append(preds.reshape(-1))
        self.targets.append(targets.reshape(-1))
        self.identities.append(identities)

    def compute(self):
        """
        Computes the metrics using the accumulated data.

        Returns:
            dict: Dictionary containing 'score', 'overall_auc', and bias submetrics.
        """
        if not self.predictions:
            return {}

        # Concatenate all batches
        all_preds = np.concatenate(self.predictions)
        all_targets = np.concatenate(self.targets)
        all_identities = np.concatenate(self.identities, axis=0)

        # Construct DataFrame expected by library.utils.calculate_jigsaw_metrics
        data = {Config.TARGET_COL: all_targets, "prediction": all_preds}

        # Add identity columns
        # The order of columns in all_identities corresponds to Config.IDENTITY_COLUMNS
        # as defined in library.data._process_and_cache (which loads them in this order)
        for idx, col_name in enumerate(Config.IDENTITY_COLUMNS):
            if idx < all_identities.shape[1]:
                data[col_name] = all_identities[:, idx]
            else:
                # Fallback if dimensions mismatch (safety check)
                data[col_name] = np.zeros_like(all_targets)

        val_df = pd.DataFrame(data)

        # Calculate metrics using the provided utility function
        metrics = calculate_jigsaw_metrics(
            val_df,
            prediction_col="prediction",
            target_col=Config.TARGET_COL,
            identity_columns=Config.IDENTITY_COLUMNS,
        )

        return metrics

    def reset(self):
        """Resets the internal storage."""
        self.predictions = []
        self.targets = []
        self.identities = []
