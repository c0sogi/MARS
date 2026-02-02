import numpy as np
import os
from sklearn.metrics import matthews_corrcoef, confusion_matrix
from library.utils import setup_logger
from library.config import IDEA_DIR


class Evaluator:
    """
    Handles evaluation metrics and threshold optimization for the RKS-MTE strategy.
    """

    def __init__(self):
        self.logger = setup_logger("evaluator")
        self.best_threshold_path = os.path.join(IDEA_DIR, "best_threshold.npy")

    def calculate_mcc(self, y_true, y_pred):
        """
        Calculates the Matthews Correlation Coefficient.

        Args:
            y_true (np.ndarray): Ground truth binary labels.
            y_pred (np.ndarray): Predicted binary labels.

        Returns:
            float: The MCC score.
        """
        # Ensure inputs are numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        return matthews_corrcoef(y_true, y_pred)

    def optimize_threshold(self, y_true, y_pred_proba, steps=100):
        """
        Finds the decision threshold that maximizes MCC.

        Args:
            y_true (np.ndarray): Ground truth binary labels.
            y_pred_proba (np.ndarray): Predicted probabilities (0 to 1).
            steps (int): Number of threshold steps to evaluate.

        Returns:
            tuple: (best_threshold, best_mcc_score)
        """
        self.logger.info(f"Optimizing threshold over {steps} steps...")

        y_true = np.array(y_true)
        y_pred_proba = np.array(y_pred_proba)

        best_threshold = 0.5
        best_score = -1.0

        # Generate thresholds to test
        # We avoid 0.0 and 1.0 to prevent trivial all-0 or all-1 predictions if possible,
        # though MCC handles them (returns 0).
        thresholds = np.linspace(0.01, 0.99, steps)

        for thresh in thresholds:
            # Binarize predictions
            y_pred_bin = (y_pred_proba >= thresh).astype(int)

            # Calculate MCC
            score = self.calculate_mcc(y_true, y_pred_bin)

            if score > best_score:
                best_score = score
                best_threshold = thresh

        self.logger.info(f"Optimization Complete.")
        self.logger.info(f"Best Threshold: {best_threshold}")
        self.logger.info(f"Best MCC Score: {best_score}")

        # Save best threshold for inference usage
        np.save(self.best_threshold_path, np.array([best_threshold]))
        self.logger.info(f"Saved best threshold to {self.best_threshold_path}")

        return best_threshold, best_score

    def print_detailed_metrics(self, y_true, y_pred):
        """
        Prints confusion matrix and detailed breakdown.
        """
        mcc = self.calculate_mcc(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        self.logger.info("Detailed Metrics:")
        self.logger.info(f"MCC: {mcc}")
        self.logger.info(f"True Negatives: {tn}")
        self.logger.info(f"False Positives: {fp}")
        self.logger.info(f"False Negatives: {fn}")
        self.logger.info(f"True Positives: {tp}")
