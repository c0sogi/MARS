import pandas as pd
import numpy as np
import os
from sklearn.metrics import matthews_corrcoef
from library.config import (
    WORKING_DIR,
    SUBMISSION_PATH,
    BEST_THRESHOLD_PATH,
    SEED,
    N_JOBS,
)
from library.features import generate_features
from library.models import LGBMClassifierWrapper, XGBClassifierWrapper


class InferencePipeline:
    """
    Manages the inference pipeline: feature generation, model loading,
    ensemble prediction, threshold optimization, and submission generation.
    """

    def __init__(self):
        self.models_dir = os.path.join(WORKING_DIR, "models")
        self.metadata_cols = [
            "contact_id",
            "game_play",
            "step",
            "nfl_player_id_1",
            "nfl_player_id_2",
            "contact",
            "datetime",
        ]
        self.lgbm_model = None
        self.xgb_model = None

    def _split_X(self, df):
        """Separates features from metadata columns."""
        feature_cols = [c for c in df.columns if c not in self.metadata_cols]
        return df[feature_cols]

    def generate_test_features(self, load_cached_data=True):
        """
        Generates or loads features for the test set.

        Args:
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            pd.DataFrame: The test dataframe with features.
        """
        print("Generating test features...")
        return generate_features(split="test", load_cached_data=load_cached_data)

    def load_models(self):
        """Loads the trained Expert models from disk."""
        lgbm_path = os.path.join(self.models_dir, "expert_lgbm.joblib")
        xgb_path = os.path.join(self.models_dir, "expert_xgb.joblib")

        if not os.path.exists(lgbm_path) or not os.path.exists(xgb_path):
            raise FileNotFoundError(
                "Trained models not found. Ensure training is complete."
            )

        print(f"Loading LightGBM model from {lgbm_path}...")
        self.lgbm_model = LGBMClassifierWrapper.load(lgbm_path)

        print(f"Loading XGBoost model from {xgb_path}...")
        self.xgb_model = XGBClassifierWrapper.load(xgb_path)

    def optimize_threshold(self, y_true, y_prob):
        """
        Finds the best threshold maximizing MCC on validation data.

        Args:
            y_true (np.array): Ground truth labels.
            y_prob (np.array): Predicted probabilities.

        Returns:
            float: The optimal threshold.
        """
        best_mcc = -1.0
        best_thresh = 0.5

        # Search space: 0.1 to 0.9
        thresholds = np.linspace(0.1, 0.9, 81)

        for thresh in thresholds:
            y_pred = (y_prob > thresh).astype(int)
            mcc = matthews_corrcoef(y_true, y_pred)
            if mcc > best_mcc:
                best_mcc = mcc
                best_thresh = thresh

        print(f"Optimized Threshold: {best_thresh}")
        print(f"Max Validation MCC: {best_mcc}")
        return best_thresh

    def predict_ensemble(self, df):
        """
        Generates ensemble probabilities for the provided dataframe.

        Args:
            df (pd.DataFrame): Dataframe containing features.

        Returns:
            np.ndarray: Averaged probabilities from LGBM and XGB.
        """
        if self.lgbm_model is None or self.xgb_model is None:
            self.load_models()

        X = self._split_X(df)

        print("Predicting with LightGBM...")
        p_lgbm = self.lgbm_model.predict_proba(X)

        print("Predicting with XGBoost...")
        p_xgb = self.xgb_model.predict_proba(X)

        # Unweighted Average
        p_ens = (p_lgbm + p_xgb) / 2.0
        return p_ens

    def get_best_threshold(self, use_saved=True, validation_df=None):
        """
        Retrieves the decision threshold. Prioritizes saved threshold.
        If not found and validation_df is provided, calculates it.

        Args:
            use_saved (bool): Whether to look for the saved threshold file.
            validation_df (pd.DataFrame): Optional validation data for re-optimization.

        Returns:
            float: The threshold to use.
        """
        # 1. Try loading saved threshold
        if use_saved and os.path.exists(BEST_THRESHOLD_PATH):
            print(f"Loading best threshold from {BEST_THRESHOLD_PATH}...")
            return float(np.load(BEST_THRESHOLD_PATH)[0])

        # 2. Optimize if validation data provided
        if validation_df is not None:
            print("Saved threshold not found. Optimizing on validation set...")
            y_val = validation_df["contact"].values
            p_val = self.predict_ensemble(validation_df)
            return self.optimize_threshold(y_val, p_val)

        # 3. Fallback
        print("Warning: No saved threshold or validation data. Defaulting to 0.5.")
        return 0.5

    def make_submission(self, test_df, probs, threshold, output_path=SUBMISSION_PATH):
        """
        Applies threshold and saves the submission file.

        Args:
            test_df (pd.DataFrame): The test dataframe (must contain contact_id).
            probs (np.ndarray): The predicted probabilities.
            threshold (float): The decision threshold.
            output_path (str): Path to save the CSV.
        """
        print(f"Applying threshold {threshold} to generate predictions...")
        predictions = (probs > threshold).astype(int)

        submission = test_df[["contact_id"]].copy()
        submission["contact"] = predictions

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        submission.to_csv(output_path, index=False)

        print(f"Submission saved to {output_path}")
        print(f"Submission shape: {submission.shape}")

    def run(self, load_cached_features=True):
        """
        Executes the full inference workflow.
        """
        # 1. Generate Test Features
        df_test = self.generate_test_features(load_cached_data=load_cached_features)

        # 2. Predict
        probs = self.predict_ensemble(df_test)

        # 3. Get Threshold
        # We assume training has run and saved the threshold.
        # If strictly inference-only without artifacts, one would need validation data.
        threshold = self.get_best_threshold(use_saved=True)

        # 4. Create Submission
        self.make_submission(df_test, probs, threshold)


def run_inference(load_cached_features=True):
    """Wrapper function to run the pipeline."""
    pipeline = InferencePipeline()
    pipeline.run(load_cached_features=load_cached_features)
