import numpy as np
import pandas as pd
import os
from sklearn.metrics import matthews_corrcoef
from library.config import ProjectConfig
from library.utils import get_logger

logger = get_logger("MetricOptimizer")


class MetricOptimizer:
    """
    Handles the evaluation of model predictions, optimization of decision thresholds
    based on the Matthews Correlation Coefficient (MCC), and generation of the
    final submission file.
    """

    def __init__(self):
        self.config = ProjectConfig
        self.submission_dir = self.config.SUBMISSION_DIR
        os.makedirs(self.submission_dir, exist_ok=True)

    def find_optimal_threshold(
        self, y_true: np.ndarray, y_probs: np.ndarray, stream_name: str
    ) -> float:
        """
        Performs a linear search to find the probability threshold that maximizes
        the Matthews Correlation Coefficient (MCC).

        Args:
            y_true (np.ndarray): Ground truth binary labels.
            y_probs (np.ndarray): Predicted probabilities.
            stream_name (str): Name of the stream ('A' or 'B') for logging.

        Returns:
            float: The optimal threshold value.
        """
        logger.info(f"Optimizing threshold for Stream {stream_name}...")

        # Define search space: 0.01 to 0.99 with step 0.01
        thresholds = np.linspace(0.01, 0.99, 99)
        best_threshold = 0.5
        best_mcc = -1.0

        # Ensure inputs are numpy arrays
        y_true = np.array(y_true)
        y_probs = np.array(y_probs)

        # Linear search
        for thresh in thresholds:
            # Convert probabilities to binary predictions
            y_pred = (y_probs >= thresh).astype(int)

            # Calculate MCC
            # Note: matthews_corrcoef handles the denominator 0 case internally
            # (returns 0), but we rely on sklearn's robust implementation.
            mcc = matthews_corrcoef(y_true, y_pred)

            if mcc > best_mcc:
                best_mcc = mcc
                best_threshold = thresh

        # Log the result with full precision
        logger.info(f"Stream {stream_name} Optimization Results:")
        logger.info(f"Best Threshold: {best_threshold}")
        logger.info(f"Best MCC: {best_mcc:.16f}")

        return best_threshold

    def generate_submission(
        self,
        probs_a: np.ndarray,
        ids_a: pd.DataFrame,
        probs_b: np.ndarray,
        ids_b: pd.DataFrame,
        thresholds: dict,
    ):
        """
        Applies optimized thresholds to test probabilities, combines streams,
        and saves the submission file.

        Args:
            probs_a (np.ndarray): Probabilities for Stream A (Interaction).
            ids_a (pd.DataFrame): Identifiers for Stream A.
            probs_b (np.ndarray): Probabilities for Stream B (Impact).
            ids_b (pd.DataFrame): Identifiers for Stream B.
            thresholds (dict): Dictionary containing 'A' and 'B' thresholds.
        """
        logger.info("Generating final submission...")

        # --- Process Stream A ---
        thresh_a = thresholds.get("A", 0.5)
        preds_a = (probs_a >= thresh_a).astype(int)

        df_a = ids_a.copy()
        df_a["contact"] = preds_a

        # --- Process Stream B ---
        thresh_b = thresholds.get("B", 0.5)
        preds_b = (probs_b >= thresh_b).astype(int)

        df_b = ids_b.copy()
        df_b["contact"] = preds_b

        # --- Combine Streams ---
        # Concatenate the results from both streams
        df_submission = pd.concat([df_a, df_b], axis=0, ignore_index=True)

        # Keep only required columns
        submission_cols = ["contact_id", "contact"]

        # Ensure we have the required columns
        if not all(col in df_submission.columns for col in submission_cols):
            raise ValueError(
                f"Missing columns for submission. Available: {df_submission.columns}"
            )

        df_submission = df_submission[submission_cols]

        # --- Save ---
        save_path = self.config.SUBMISSION_PATH
        df_submission.to_csv(save_path, index=False)

        logger.info(f"Submission saved to {save_path}")
        logger.info(f"Total predictions: {len(df_submission)}")
        logger.info(f"Positive predictions: {df_submission['contact'].sum()}")
