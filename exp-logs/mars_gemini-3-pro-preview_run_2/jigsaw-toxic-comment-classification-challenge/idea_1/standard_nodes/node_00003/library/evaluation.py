import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.utils import setup_logger


def compute_score(y_true, y_pred):
    """
    Computes the Mean Column-wise ROC AUC score for the toxic comment classification task.

    Args:
        y_true (pd.DataFrame): The ground truth labels. Must contain columns specified in Config.TARGET_COLS.
        y_pred (pd.DataFrame): The predicted probabilities. Must contain columns specified in Config.TARGET_COLS.

    Returns:
        float: The mean column-wise ROC AUC score.
    """
    logger = setup_logger("evaluation")
    target_cols = Config.TARGET_COLS

    logger.info("=== Starting Evaluation ===")

    scores = []

    # Iterate over each target label to calculate individual AUC
    for label in target_cols:
        # Verify columns exist
        if label not in y_true.columns:
            raise ValueError(f"Label '{label}' not found in y_true columns.")
        if label not in y_pred.columns:
            raise ValueError(f"Label '{label}' not found in y_pred columns.")

        # Extract the specific column vectors
        y_t = y_true[label]
        y_p = y_pred[label]

        try:
            # Calculate ROC AUC for this label
            score = roc_auc_score(y_t, y_p)
        except ValueError as e:
            # This can happen if y_true has only one class (e.g., all 0s) in a small batch
            logger.warning(f"Error calculating AUC for label '{label}': {e}")
            score = 0.5  # Fallback for undefined AUC

        scores.append(score)

        # Log the individual score with full precision
        logger.info(f"Label: {label} - AUC: {score}")

    # Calculate the mean of the individual AUC scores
    mean_auc = np.mean(scores)

    logger.info(f"Mean Column-wise ROC AUC: {mean_auc}")
    logger.info("=== Evaluation Complete ===")

    return mean_auc
