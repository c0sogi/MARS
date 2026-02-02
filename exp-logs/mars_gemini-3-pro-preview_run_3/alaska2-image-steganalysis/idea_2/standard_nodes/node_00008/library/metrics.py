import numpy as np
from library.utils import weighted_auc


def alaska_weighted_auc(y_true, y_pred):
    """
    Calculates the Weighted AUC metric for the Alaska2 Steganalysis task.

    This metric computes the area under the ROC curve, but weights specific regions
    of the True Positive Rate (TPR) differently.

    Weights configuration (from Config):
    - TPR range [0.0, 0.4]: Weighted 2x
    - TPR range [0.4, 1.0]: Weighted 1x

    The final score is normalized to be between 0 and 1.

    Args:
        y_true (array-like): Ground truth binary labels (0 for Cover, 1 for Stego).
        y_pred (array-like): Predicted probabilities or logits. Higher values indicate
                             higher confidence in the 'Stego' class.

    Returns:
        float: The weighted AUC score.
    """
    # Delegate the calculation to the centralized utility function
    # to ensure consistency with the global configuration (Config.TPR_THRESHOLDS, etc.)
    return weighted_auc(y_true, y_pred)
