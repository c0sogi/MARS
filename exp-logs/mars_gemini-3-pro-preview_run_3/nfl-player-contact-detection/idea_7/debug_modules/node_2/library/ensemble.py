import numpy as np
import pandas as pd
import os
from library.config import Config
from library.utils import compute_mcc


class EnsembleOptimizer:
    """
    Handles the post-processing and fusion logic for the Multi-Modal Late-Fusion Ensemble.
    Optimizes blending weights and decision thresholds to maximize MCC.
    """

    def __init__(self):
        """
        Initializes the optimizer with global configuration.
        """
        self.config = Config

    def blend_predictions(
        self, pred_a: np.ndarray, pred_b: np.ndarray, weight_a: float
    ) -> np.ndarray:
        """
        Combines predictions from Stream A (Tracking) and Stream B (Helmets) using a linear weighted average.

        Args:
            pred_a (np.ndarray): Probability predictions from Stream A.
            pred_b (np.ndarray): Probability predictions from Stream B.
            weight_a (float): Weight assigned to Stream A (0.0 to 1.0). Stream B gets (1 - weight_a).

        Returns:
            np.ndarray: Blended probability array.
        """
        # Ensure inputs are numpy arrays
        p_a = np.asarray(pred_a)
        p_b = np.asarray(pred_b)

        # Linear blend: P_final = w_A * P_A + (1 - w_A) * P_B
        return weight_a * p_a + (1.0 - weight_a) * p_b

    def optimize_threshold(self, y_true: np.ndarray, blended_probs: np.ndarray):
        """
        Performs a linear search to find the optimal decision threshold that maximizes MCC.

        Args:
            y_true (np.ndarray): Ground truth binary labels.
            blended_probs (np.ndarray): Blended probability predictions.

        Returns:
            tuple: (best_threshold, best_mcc)
        """
        best_threshold = 0.5
        best_mcc = -1.0

        # Iterate over thresholds defined in Config
        for threshold in self.config.THRESHOLDS:
            # Binarize predictions based on current threshold
            y_pred = (blended_probs >= threshold).astype(int)

            # Calculate MCC
            score = compute_mcc(y_true, y_pred)

            if score > best_mcc:
                best_mcc = score
                best_threshold = threshold

        return best_threshold, best_mcc

    def optimize_weights(
        self, y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray
    ):
        """
        Performs a grid search to find the best blending weight for Stream A.
        For each weight candidate, it implicitly finds the best threshold to evaluate the potential MCC.

        Args:
            y_true (np.ndarray): Ground truth binary labels.
            pred_a (np.ndarray): Probability predictions from Stream A.
            pred_b (np.ndarray): Probability predictions from Stream B.

        Returns:
            float: The optimal weight for Stream A.
        """
        best_weight = 0.5
        global_best_mcc = -1.0

        print("Starting Weight Optimization...")

        # Iterate over weights defined in Config
        for weight in self.config.BLEND_WEIGHTS:
            # Blend predictions with current weight
            blended = self.blend_predictions(pred_a, pred_b, weight)

            # Find the best threshold (and resulting MCC) for this specific blend
            _, current_mcc = self.optimize_threshold(y_true, blended)

            # Update global best if this weight yields a better result
            if current_mcc > global_best_mcc:
                global_best_mcc = current_mcc
                best_weight = weight

        return best_weight

    def optimize(self, y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray):
        """
        Convenience method to perform full optimization (weights and threshold) in one step.
        Prints the results with full precision.

        Args:
            y_true (np.ndarray): Ground truth binary labels.
            pred_a (np.ndarray): Probability predictions from Stream A.
            pred_b (np.ndarray): Probability predictions from Stream B.

        Returns:
            tuple: (best_weight, best_threshold, best_mcc)
        """
        # 1. Find optimal weight
        best_weight = self.optimize_weights(y_true, pred_a, pred_b)

        # 2. Re-calculate blended probs with best weight
        best_blend = self.blend_predictions(pred_a, pred_b, best_weight)

        # 3. Find optimal threshold for this best blend
        best_threshold, best_mcc = self.optimize_threshold(y_true, best_blend)

        print(f"Optimization Results:")
        print(f"Best Weight (Stream A): {best_weight}")
        print(f"Best Threshold: {best_threshold}")
        print(f"Validation MCC: {best_mcc}")

        return best_weight, best_threshold, best_mcc

    def save_submission(self, submission_df: pd.DataFrame, output_path: str = None):
        """
        Saves the submission DataFrame to a CSV file.

        Args:
            submission_df (pd.DataFrame): The dataframe containing contact_id and contact predictions.
            output_path (str, optional): Path to save the file. Defaults to Config.SUBMISSION_PATH.
        """
        if output_path is None:
            output_path = self.config.SUBMISSION_PATH

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        print(f"Saving submission to {output_path}...")
        submission_df.to_csv(output_path, index=False)
