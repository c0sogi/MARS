import os
import numpy as np
import pandas as pd
import joblib
from library.config import Config
from library.utils import setup_logger
from library.feature_engineering import FeatureEngineer


class InferenceManager:
    """
    Manages the inference process for the HAMF-ADBE model.

    This class handles loading the test data, restoring the trained ensemble models
    (and their specific preprocessors), generating predictions via CV-Bagging,
    and saving the final submission file.
    """

    def __init__(self):
        self.logger = setup_logger("inference_manager")
        self.feature_engineer = FeatureEngineer()

    def predict_test_set(self, load_cached_data: bool = True):
        """
        Generates predictions for the test set using the ensemble of trained fold models.

        Args:
            load_cached_data (bool): Whether to use cached features/embeddings during
                                     data loading via FeatureEngineer.
        """
        self.logger.info("Starting inference pipeline...")

        # 1. Load Feature Set
        # We use the FeatureEngineer to retrieve the test features.
        # This handles embedding loading/computation and metadata extraction.
        features = self.feature_engineer.build_feature_set(
            load_cached_data=load_cached_data
        )

        if "test" not in features:
            raise ValueError("Feature set does not contain 'test' data.")

        test_features = features["test"]
        request_ids = test_features["request_id"]

        # 2. Prepare Data Dictionary for Preprocessor
        # The HAMFPreprocessor expects a dictionary with specific keys corresponding to the views.
        valid_keys = [
            "anchor_title",
            "anchor_body",
            "aux_global",
            "aux_hook",
            "metadata",
        ]

        # Filter test_features to only include valid input keys
        X_test_dict = {k: test_features[k] for k in valid_keys if k in test_features}

        # Verify all required keys are present to prevent runtime errors during transformation
        for key in valid_keys:
            if key not in X_test_dict:
                raise ValueError(f"Missing required feature key in test set: {key}")

        fold_preds = []

        # 3. Iterate over folds to generate predictions
        self.logger.info(f"Aggregating predictions across {Config.N_FOLDS} folds...")

        for fold in range(Config.N_FOLDS):
            # Define paths for the model and preprocessor artifacts
            model_path = os.path.join(
                Config.MODEL_CHECKPOINT_DIR, f"model_fold_{fold}.joblib"
            )
            proc_path = os.path.join(
                Config.MODEL_CHECKPOINT_DIR, f"processor_fold_{fold}.joblib"
            )

            # Validate artifact existence
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Model file not found for fold {fold}: {model_path}"
                )
            if not os.path.exists(proc_path):
                raise FileNotFoundError(
                    f"Preprocessor file not found for fold {fold}: {proc_path}"
                )

            try:
                self.logger.info(f"Processing fold {fold}...")

                # Load artifacts
                model = joblib.load(model_path)
                preprocessor = joblib.load(proc_path)

                # Transform test data using the fold-specific preprocessor
                # This applies the fold-specific PCA projections and scalers
                X_test_processed = preprocessor.transform(X_test_dict)

                # Predict probabilities (Target class 1: Received Pizza)
                preds = model.predict_proba(X_test_processed)[:, 1]
                fold_preds.append(preds)

            except Exception as e:
                self.logger.error(f"Error during inference for fold {fold}: {e}")
                raise e

        if not fold_preds:
            raise RuntimeError("No predictions were generated. Check model artifacts.")

        # 4. CV-Bagging: Average predictions across all folds
        avg_preds = np.mean(fold_preds, axis=0)

        # 5. Save Submission
        self.logger.info("Generating submission DataFrame...")
        df_sub = pd.DataFrame(
            {
                "request_id": request_ids,
                "requester_received_pizza": avg_preds,
            }
        )

        # Ensure submission directory exists
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        self.logger.info(f"Submission saved successfully to {Config.SUBMISSION_PATH}")
