import os
import joblib
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import setup_logger
from library.data_loader import DataLoader
from library.feature_engineering import Embedder, ViewTransformer


class Predictor:
    """
    Handles inference on the test set for Idea 17: AMBLE.
    Loads trained models and transformers from all folds, performs inference,
    aggregates predictions, and generates the submission file.
    """

    def __init__(self):
        self.logger = setup_logger("Predictor")
        self.data_loader = DataLoader()
        self.embedder = Embedder()
        self.models_dir = os.path.join(Config.WORKING_DIR, "models")

    def generate_submission(self, sample_size=None):
        """
        Generates predictions for the test set using the ensemble of trained fold models.
        Averages the probabilities (Bagging) and saves to CSV.
        """
        self.logger.info("Starting submission generation...")

        # 1. Load Test Data (Tabular)
        df_test = self.data_loader.load_dataset(split="test", load_cached_data=True)

        # 2. Load Embeddings
        X_req = self.embedder.get_embeddings(
            df_test, split="test", view_type="request", load_cached_data=True
        )
        X_hist = self.embedder.get_embeddings(
            df_test, split="test", view_type="history", load_cached_data=True
        )

        # 3. Extract Metadata
        meta_cols = self.data_loader.numerical_features
        X_meta = df_test[meta_cols].values

        # Concatenate
        X_all = np.hstack([X_req, X_hist, X_meta])
        ids = df_test["request_id"].values

        # 4. Handle Subsetting
        if sample_size is not None:
            self.logger.info(f"Subsetting test data to first {sample_size} samples.")
            X_all = X_all[:sample_size]
            ids = ids[:sample_size]

        # 5. Inference Loop
        test_preds_sum = np.zeros(len(ids))
        models_found = 0

        for fold_idx in range(Config.N_FOLDS):
            model_path = os.path.join(
                self.models_dir, f"pipeline_fold_{fold_idx}.joblib"
            )

            if not os.path.exists(model_path):
                self.logger.warning(
                    f"Pipeline for Fold {fold_idx} not found at {model_path}. Skipping."
                )
                continue

            try:
                pipeline = joblib.load(model_path)
            except Exception as e:
                self.logger.error(f"Failed to load pipeline for Fold {fold_idx}: {e}")
                continue

            # Predict Probability (Pipeline handles transformation)
            probs = pipeline.predict_proba(X_all)[:, 1]
            test_preds_sum += probs
            models_found += 1

            self.logger.info(f"Processed inference for Fold {fold_idx}.")

        if models_found == 0:
            raise RuntimeError("No trained models found. Cannot generate submission.")

        # Average predictions
        avg_preds = test_preds_sum / models_found

        # 6. Create Submission DataFrame
        submission_df = pd.DataFrame(
            {"request_id": ids, "requester_received_pizza": avg_preds}
        )

        # 7. Save Submission
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        self.logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
