import os
import numpy as np
import pandas as pd
from library.config import KADM_CONFIG
from library.utils import setup_logger, calc_mcc, save_submission
from library.model_zoo import DualModelEnsemble
from library.data_loader import DataLoader

# Setup logger
logger = setup_logger(name="inference_engine")


class ThresholdOptimizer:
    """
    Optimizes the decision threshold on the validation set to maximize MCC.
    """

    def __init__(self, config=KADM_CONFIG):
        self.config = config
        self.model_dir = config["paths"]["model_dir"]

    def optimize(self, ensemble, X_val, y_val):
        """
        Finds the optimal threshold.

        Args:
            ensemble (DualModelEnsemble): The trained model ensemble.
            X_val (pd.DataFrame): Validation features.
            y_val (pd.Series): Validation targets.

        Returns:
            float: The optimal threshold.
        """
        logger.info("Starting threshold optimization on validation set...")

        # Generate raw probabilities
        probs = ensemble.predict(X_val)

        # Grid search for threshold
        thresholds = np.arange(0.01, 1.00, 0.01)
        best_mcc = -1.0
        best_threshold = 0.5

        for thresh in thresholds:
            preds_bin = (probs > thresh).astype(int)
            mcc = calc_mcc(y_val, preds_bin)

            if mcc > best_mcc:
                best_mcc = mcc
                best_threshold = thresh

        logger.info(f"Optimization Complete. Best MCC: {best_mcc}")
        logger.info(f"Optimal Threshold: {best_threshold}")

        # Save threshold
        thresh_path = os.path.join(self.model_dir, "best_threshold.npy")
        np.save(thresh_path, np.array([best_threshold]))
        logger.info(f"Threshold saved to {thresh_path}")

        return best_threshold


class Predictor:
    """
    Handles inference on the test set, applying gating logic and generating final predictions.
    """

    def __init__(self, config=KADM_CONFIG):
        self.config = config
        self.data_loader = DataLoader(config)

    def generate_test_predictions(self, ensemble, threshold, load_cached_data=True):
        """
        Generates predictions for the test set.

        Args:
            ensemble (DualModelEnsemble): Trained ensemble.
            threshold (float): Decision threshold.
            load_cached_data (bool): Whether to use cached features.

        Returns:
            pd.DataFrame: DataFrame with 'contact_id' and 'contact' (prediction).
        """
        logger.info("Generating test predictions...")

        # 1. Load Test Data
        # Crucial: We set apply_gating=False because we need to return predictions
        # for ALL rows in the sample submission. We will use the 'gating_pass' column
        # to force 0 on rows that failed gating.
        X_test, _, meta_info = self.data_loader.load_dataset(
            "test", apply_gating=False, load_cached_data=load_cached_data
        )

        # 2. Identify Gated Candidates
        # 'gating_pass' is a boolean column in meta_info indicating if the pair passed the quadratic filter
        if "gating_pass" in meta_info.columns:
            gating_mask = meta_info["gating_pass"].astype(bool)
        else:
            logger.warning(
                "'gating_pass' not found in metadata. Predicting for all rows."
            )
            gating_mask = pd.Series(True, index=X_test.index)

        # 3. Inference
        # We only strictly need to predict on rows where gating_mask is True to save time,
        # but for simplicity and vectorization, we can predict all or mask.
        # Let's predict all and then zero out.
        raw_probs = ensemble.predict(X_test)

        # 4. Apply Gating Logic (Force 0 for filtered candidates)
        # If the candidate didn't pass the distance/physics check, prob is 0.
        final_probs = raw_probs.copy()
        final_probs[~gating_mask] = 0.0

        # 5. Apply Threshold
        predictions = (final_probs > threshold).astype(int)

        # 6. Format Output
        submission_df = pd.DataFrame(
            {"contact_id": meta_info["contact_id"], "contact": predictions}
        )

        return submission_df


def generate_submission(
    ensemble=None, threshold=None, load_cached_data=True, config=KADM_CONFIG
):
    """
    Orchestrates the submission generation process.

    Args:
        ensemble (DualModelEnsemble, optional): Loaded model. If None, loads from disk.
        threshold (float, optional): Decision threshold. If None, loads from disk.
        load_cached_data (bool): Usage of cached features.
        config (dict): Configuration dictionary.
    """
    logger.info("Starting submission generation pipeline...")

    # 1. Load Model if not provided
    if ensemble is None:
        logger.info("Loading ensemble from disk...")
        ensemble = DualModelEnsemble(config)
        ensemble.load()

    # 2. Load Threshold if not provided
    if threshold is None:
        thresh_path = os.path.join(config["paths"]["model_dir"], "best_threshold.npy")
        if os.path.exists(thresh_path):
            threshold = float(np.load(thresh_path)[0])
            logger.info(f"Loaded threshold from disk: {threshold}")
        else:
            logger.warning("Threshold file not found. Defaulting to 0.5.")
            threshold = 0.5

    # 3. Generate Predictions
    predictor = Predictor(config)
    submission_df = predictor.generate_test_predictions(
        ensemble, threshold, load_cached_data=load_cached_data
    )

    # 4. Save Submission
    save_submission(submission_df)

    logger.info("Submission generation completed successfully.")
