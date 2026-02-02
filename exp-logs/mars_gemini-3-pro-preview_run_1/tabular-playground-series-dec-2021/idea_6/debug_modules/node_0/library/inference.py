import pandas as pd
import numpy as np
import xgboost as xgb
import os
from library import config


class InferenceEngine:
    """
    Handles prediction generation and submission formatting using a trained ensemble.
    """

    def __init__(self, models, label_encoder, feature_names=None):
        """
        Initializes the InferenceEngine.

        Args:
            models (list): List of trained XGBoost Booster models.
            label_encoder (LabelEncoder): The fitted LabelEncoder used during training.
            feature_names (list, optional): List of feature names to ensure column alignment.
        """
        self.models = models
        self.le = label_encoder
        self.feature_names = feature_names

    def predict_ensemble(self, df_test):
        """
        Generates soft probability predictions for the test set by averaging
        outputs from all trained models (Soft Voting).

        Args:
            df_test (pd.DataFrame): The processed test dataset.

        Returns:
            np.ndarray: The averaged probability matrix (n_samples, n_classes).
        """
        if not self.models:
            raise ValueError("No models available for inference.")

        # Prepare Test Data
        # Drop ID column if present
        X_test = df_test.drop(columns=[config.ID_COL], errors="ignore")

        # Align columns to match training features
        if self.feature_names:
            # Ensure all required features are present
            missing_cols = set(self.feature_names) - set(X_test.columns)
            if missing_cols:
                raise ValueError(f"Test data is missing features: {missing_cols}")

            # Reorder columns to match training data
            X_test = X_test[self.feature_names]

        # Create DMatrix for XGBoost
        dtest = xgb.DMatrix(X_test)

        # Initialize array for accumulating probabilities
        num_classes = len(self.le.classes_)
        avg_preds = np.zeros((len(df_test), num_classes), dtype=np.float32)

        print(f"Generating predictions using {len(self.models)} models...")

        # Soft Voting: Accumulate probabilities from each model
        for i, model in enumerate(self.models):
            preds = model.predict(dtest)
            avg_preds += preds

        # Compute average
        avg_preds /= len(self.models)

        return avg_preds

    def save_submission(self, df_test, probabilities):
        """
        Maps the averaged probabilities to class labels and writes the final submission CSV file.

        Args:
            df_test (pd.DataFrame): The test dataset (used to retrieve Ids).
            probabilities (np.ndarray): The averaged probability matrix.

        Returns:
            pd.DataFrame: The submission dataframe.
        """
        if config.ID_COL not in df_test.columns:
            raise ValueError(f"Test dataframe missing '{config.ID_COL}' column.")

        # Determine class indices (argmax of probabilities)
        class_indices = np.argmax(probabilities, axis=1)

        # Inverse transform to original class labels
        final_predictions = self.le.inverse_transform(class_indices)

        # Create submission DataFrame
        submission = pd.DataFrame(
            {
                config.ID_COL: df_test[config.ID_COL],
                config.TARGET_COL: final_predictions,
            }
        )

        # Ensure submission directory exists
        os.makedirs(config.SUBMISSION_DIR, exist_ok=True)

        # Save to CSV
        submission.to_csv(config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {config.SUBMISSION_PATH}")

        return submission
