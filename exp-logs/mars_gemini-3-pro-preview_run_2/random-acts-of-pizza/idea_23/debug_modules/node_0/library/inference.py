import os
import numpy as np
import pandas as pd
import joblib
from library.config import Config
from library.utils import setup_logger
from library.data_loader import DataLoader
from library.feature_generator import FeatureGenerator
from library.custom_ensemble import StratifiedRandomSubspaceEnsemble


class InferenceRunner:
    """
    Manages the inference process for the Stratified Random Subspace Linear Ensemble.
    Loads trained models and scalers from all folds, generates predictions,
    and creates the final submission file.
    """

    def __init__(self):
        self.logger = setup_logger("InferenceRunner")
        self.data_loader = DataLoader()
        self.feature_gen = FeatureGenerator()

    def generate_submission(self):
        """
        Executes the full inference pipeline:
        1. Load test data and features.
        2. Load models and scalers for each fold.
        3. Generate predictions (CV-Bagging).
        4. Save submission file.
        """
        self.logger.info("Starting Inference Pipeline...")

        # 1. Load Test Data
        # We only need the test split here. load_data returns (train, val, test)
        _, _, df_test = self.data_loader.load_data(load_cached_data=True)

        # 2. Generate Features
        # Text Embeddings (Cached)
        X_text_test = self.feature_gen.generate_embeddings(
            df_test, split_name="test", load_cached_data=True
        )

        # Tabular Metadata (Raw)
        X_tab_test_raw = self.feature_gen.extract_tabular_features(df_test)

        # 3. CV-Bagging Inference
        # We will accumulate predictions from each fold
        n_samples = len(df_test)
        fold_preds = np.zeros((n_samples, Config.N_FOLDS))

        models_dir = os.path.join(Config.WORKING_DIR, "models")
        if not os.path.exists(models_dir):
            raise FileNotFoundError(
                f"Models directory not found: {models_dir}. Please run training first."
            )

        self.logger.info(f"Generating predictions using {Config.N_FOLDS} folds...")

        for fold in range(Config.N_FOLDS):
            fold_idx = fold + 1  # 1-based index for logging/filenames as per Trainer

            # Define paths
            model_path = os.path.join(models_dir, f"model_fold_{fold}.joblib")
            scaler_path = os.path.join(models_dir, f"scaler_fold_{fold}.joblib")

            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                self.logger.warning(f"Artifacts for Fold {fold} not found. Skipping...")
                continue

            # Load Artifacts
            try:
                scaler = joblib.load(scaler_path)
                model = joblib.load(model_path)
            except Exception as e:
                self.logger.error(f"Error loading artifacts for Fold {fold}: {e}")
                continue

            # Preprocess Tabular Data for this Fold
            # Use the fold-specific scaler to transform test data
            X_tab_test_scaled = scaler.transform(X_tab_test_raw)

            # Predict
            # The ensemble expects (X_text, X_tabular)
            probs = model.predict_proba(X_text_test, X_tab_test_scaled)[:, 1]
            fold_preds[:, fold] = probs

            self.logger.info(f"Fold {fold_idx} inference complete.")

        # 4. Aggregate Predictions (Average)
        avg_preds = np.mean(fold_preds, axis=1)

        # 5. Create Submission
        self._save_submission(df_test, avg_preds)

    def _save_submission(self, df_test: pd.DataFrame, preds: np.ndarray):
        """
        Formats and saves the submission file.

        Args:
            df_test (pd.DataFrame): Test dataframe containing request_ids.
            preds (np.ndarray): Predicted probabilities.
        """
        submission = pd.DataFrame(
            {Config.ID_COL: df_test[Config.ID_COL], Config.TARGET_COL: preds}
        )

        # Ensure directory exists
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

        # Save
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        self.logger.info(f"Submission saved successfully to {Config.SUBMISSION_PATH}")

        # Log preview
        self.logger.info("Submission Preview:")
        self.logger.info(f"\n{submission.head()}")
