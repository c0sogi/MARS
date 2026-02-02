import os
import glob
import joblib
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import setup_logger
from library.dataset_builder import DatasetBuilder

logger = setup_logger("predictor")


class Predictor:
    """
    Handles the inference phase of the Dual-Resolution Semantic Early Fusion strategy.
    Loads trained fold-models, generates predictions for the test set, and creates
    the final submission file via soft-voting (averaging probabilities).
    """

    def __init__(self):
        self.config = Config
        self.models_dir = os.path.join(self.config.WORKING_DIR, "models")

    def generate_submission(self, load_cached_data=True):
        """
        Generates the submission file by averaging predictions from all available trained models.

        Args:
            load_cached_data (bool): If True, attempts to load pre-processed test data from cache.
                                     If False or cache missing, triggers data processing.
        """
        logger.info("Starting submission generation process...")

        # 1. Load Test Data
        # We use DatasetBuilder to ensure consistency with training data processing
        builder = DatasetBuilder()
        # build_datasets returns: (X_train, y_train, X_val, y_val, X_test, test_ids)
        # We only need the test components
        logger.info("Retrieving test dataset...")
        _, _, _, _, X_test, test_ids = builder.build_datasets(
            load_cached_data=load_cached_data
        )

        if X_test is None or test_ids is None:
            logger.error("Failed to load test data.")
            return

        # 2. Identify Trained Models
        # Look for files matching the pattern saved by Trainer
        model_pattern = os.path.join(self.models_dir, "model_fold_*.joblib")
        model_files = sorted(glob.glob(model_pattern))

        if not model_files:
            logger.error(
                f"No trained models found in {self.models_dir}. Please train the models first."
            )
            return

        logger.info(f"Found {len(model_files)} models for ensemble inference.")

        # 3. Inference Loop (CV-Bagging)
        # Initialize accumulator for probabilities
        total_preds = np.zeros(len(X_test))

        for model_path in model_files:
            try:
                logger.info(f"Loading model: {os.path.basename(model_path)}")
                model = joblib.load(model_path)

                # Generate probabilities for the positive class (1)
                # The pipeline handles all preprocessing (scaling, PCA, normalization) internally
                preds = model.predict_proba(X_test)[:, 1]
                total_preds += preds

            except Exception as e:
                logger.error(f"Error using model {model_path}: {e}")
                # We continue to try other models, but this might affect the average
                # In a strict setting, we might want to raise, but here we proceed.

        # 4. Average Predictions
        avg_preds = total_preds / len(model_files)

        # 5. Create Submission DataFrame
        submission_df = pd.DataFrame(
            {"request_id": test_ids, "requester_received_pizza": avg_preds}
        )

        # 6. Save Submission
        os.makedirs(os.path.dirname(self.config.SUBMISSION_PATH), exist_ok=True)
        submission_df.to_csv(self.config.SUBMISSION_PATH, index=False)

        logger.info(
            f"Submission generated successfully with {len(submission_df)} rows."
        )
        logger.info(f"Saved to: {self.config.SUBMISSION_PATH}")
