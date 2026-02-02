import os
import joblib
import numpy as np
import pandas as pd
from library.config import Config
from library.data_loader import DataLoader
from library.text_encoder import TextEncoder
from library.tabular_processor import TabularProcessor


class Predictor:
    """
    Handles the inference phase of the Supervised Semantic Projection Ensemble (SSPE).
    Loads pre-trained model artifacts for each fold, processes the test data,
    and generates the final averaged submission file.
    """

    def __init__(self):
        """
        Initialize the Predictor with necessary processors and path configurations.
        """
        self.models_dir = os.path.join(Config.WORKING_DIR, "models")
        self.text_encoder = TextEncoder()
        self.tabular_processor = TabularProcessor()

    def predict_submission(self, load_cached_data: bool = True):
        """
        Generates predictions for the test set and saves the submission file.

        Args:
            load_cached_data (bool): Whether to use cached features if available.
        """
        # ==========================================
        # 1. Load Test Data
        # ==========================================
        print("Loading test data for inference...")
        # DataLoader returns (train, val, test), we only need test
        _, _, df_test = DataLoader.load_data(load_cached_data=load_cached_data)

        # ==========================================
        # 2. Extract Features
        # ==========================================
        print("Extracting test features...")

        # Text Embeddings (MPNet)
        # Uses caching defined in Config (test_embeddings.npy)
        X_text_test = self.text_encoder.encode(
            df_test, Config.TEST_EMBEDDINGS_PATH, load_cached_data
        )

        # Tabular Features
        # Processed on the fly or cached if implemented in processor
        X_tab_test = self.tabular_processor.process(df_test)

        # ==========================================
        # 3. Ensemble Inference (Bagging over Folds)
        # ==========================================
        print(f"Starting inference using {Config.N_SPLITS}-fold ensemble...")

        # Initialize array to store sum of predictions
        test_preds_sum = np.zeros(len(df_test))

        # Check if models directory exists
        if not os.path.exists(self.models_dir):
            raise FileNotFoundError(
                f"Models directory not found at {self.models_dir}. "
                "Please run training before inference."
            )

        for fold in range(Config.N_SPLITS):
            # Define paths for this fold's artifacts
            tab_scaler_path = os.path.join(
                self.models_dir, f"tab_scaler_fold_{fold}.joblib"
            )
            clf_path = os.path.join(self.models_dir, f"clf_fold_{fold}.joblib")

            # Verify artifacts exist
            if not all(os.path.exists(p) for p in [tab_scaler_path, clf_path]):
                raise FileNotFoundError(
                    f"Missing artifacts for fold {fold}. Ensure training completed successfully."
                )

            # Load artifacts
            tab_scaler = joblib.load(tab_scaler_path)
            clf = joblib.load(clf_path)

            # --- Apply Transformations ---

            # 1. Scale Tabular Features (RankGauss/QuantileTransformer)
            X_tab_scaled = tab_scaler.transform(X_tab_test)

            # 2. Feature Fusion
            X_final = np.hstack([X_text_test, X_tab_scaled])

            # --- Generate Predictions ---
            # Predict probabilities for the positive class (1)
            fold_preds = clf.predict_proba(X_final)[:, 1]

            # Add to accumulator
            test_preds_sum += fold_preds

        # ==========================================
        # 4. Aggregation and Submission
        # ==========================================
        # Average predictions across all folds
        avg_preds = test_preds_sum / Config.N_SPLITS

        # Create submission DataFrame
        submission = pd.DataFrame(
            {
                "request_id": df_test["request_id"],
                "requester_received_pizza": avg_preds,
            }
        )

        # Save to CSV
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        submission.to_csv(Config.SUBMISSION_PATH, index=False)

        print(f"Inference complete. Submission saved to {Config.SUBMISSION_PATH}")
