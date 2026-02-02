import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import matthews_corrcoef

from library.config import Config
from library.data_manager import DataManager
from library.models import LGBMWrapper, XGBWrapper, HistGBWrapper


class Evaluator:
    def __init__(self):
        self.config = Config
        self.dm = DataManager()
        self.model_wrappers = [LGBMWrapper, XGBWrapper, HistGBWrapper]
        self.models = []

        # Directories
        self.expert_dir = os.path.join(self.config.MODEL_DIR, "experts")
        self.working_dir = self.config.WORKING_DIR
        self.submission_dir = self.config.SUBMISSION_DIR

        # Ensure directories exist
        os.makedirs(self.working_dir, exist_ok=True)
        os.makedirs(self.submission_dir, exist_ok=True)

    def load_expert_models(self):
        """
        Loads the trained expert models from the expert directory.
        """
        self.models = []
        print(f"Loading expert models from {self.expert_dir}...")
        for Wrapper in self.model_wrappers:
            model = Wrapper()
            model.model_dir = self.expert_dir
            model.load()
            if model.model is not None:
                self.models.append(model)
            else:
                print(f"Warning: {model.name} could not be loaded.")

        if not self.models:
            raise RuntimeError(
                "No expert models were loaded. Ensure training has completed successfully."
            )

    def predict_ensemble(self, X):
        """
        Generates averaged probability predictions from the loaded ensemble models.

        Args:
            X (pd.DataFrame): Feature matrix.

        Returns:
            np.ndarray: Averaged probabilities of contact.
        """
        if not self.models:
            raise ValueError("Models not loaded. Call load_expert_models() first.")

        # Initialize accumulator
        preds_sum = np.zeros(len(X))

        for model in self.models:
            preds = model.predict(X)
            preds_sum += preds

        # Unweighted average
        return preds_sum / len(self.models)

    def optimize_threshold(self, load_cached_data=True):
        """
        Finds the decision threshold that maximizes MCC on the validation set.
        Caches the result to avoid re-calculation.

        Args:
            load_cached_data (bool): Whether to load the threshold from cache if available.

        Returns:
            float: The optimized threshold.
        """
        cache_path = os.path.join(self.working_dir, "best_threshold.npy")

        # 1. Check Cache
        if load_cached_data and os.path.exists(cache_path):
            best_threshold = np.load(cache_path)
            # Handle 0-d array from numpy save
            best_threshold = float(best_threshold)
            print(f"Loaded cached best threshold: {best_threshold:.16f}")
            return best_threshold

        print("Optimizing threshold on validation set...")

        # 2. Load Validation Data
        # Note: This returns the gated validation set. Optimization focuses on the "danger zone".
        df_val = self.dm.get_val_features(load_cached_data=load_cached_data)
        X_val = df_val[self.config.FEATURES]
        # Use raw 'contact' for metric evaluation, not the smoothed one
        y_true = df_val["contact"].values

        # 3. Load Models and Predict
        self.load_expert_models()
        y_probs = self.predict_ensemble(X_val)

        # 4. Grid Search for Best Threshold
        thresholds = np.linspace(0.01, 0.99, 99)
        best_mcc = -1.0
        best_threshold = 0.5

        for thresh in thresholds:
            y_pred = (y_probs >= thresh).astype(int)
            mcc = matthews_corrcoef(y_true, y_pred)

            if mcc > best_mcc:
                best_mcc = mcc
                best_threshold = thresh

        print(
            f"Optimization Complete. Max MCC: {best_mcc:.16f} at Threshold: {best_threshold:.16f}"
        )

        # 5. Save Cache
        np.save(cache_path, np.array(best_threshold))

        return float(best_threshold)

    def generate_submission(self, threshold, load_cached_data=True):
        """
        Generates the submission file.
        Predicts on gated test features, maps them to contact_ids, and fills missing (gated) rows with 0.

        Args:
            threshold (float): The decision threshold to apply.
            load_cached_data (bool): Passed to data manager for loading features.
        """
        print(f"Generating submission with threshold {threshold:.16f}...")

        # 1. Load Test Features (Gated)
        df_test_features = self.dm.get_test_features(load_cached_data=load_cached_data)

        # 2. Reconstruct contact_id
        # The features dataframe has component parts. We must reconstruct the ID to merge with sample submission.
        # Format: game_play_step_player1_player2
        # Ensure types are string for concatenation
        gp = df_test_features["game_play"].astype(str)
        step = df_test_features["step"].astype(str)
        p1 = df_test_features["nfl_player_id_1"].astype(int).astype(str)
        p2 = df_test_features["nfl_player_id_2"].astype(str)  # Can be 'G' or player ID

        df_test_features["contact_id"] = gp + "_" + step + "_" + p1 + "_" + p2

        # 3. Generate Predictions
        # Ensure models are loaded (if run separately from optimize)
        if not self.models:
            self.load_expert_models()

        X_test = df_test_features[self.config.FEATURES]
        probs = self.predict_ensemble(X_test)
        preds = (probs >= threshold).astype(int)

        # Create a dataframe of predictions for the gated survivors
        df_preds = pd.DataFrame(
            {"contact_id": df_test_features["contact_id"], "contact_pred": preds}
        )

        # 4. Merge with Sample Submission
        # Load the template to get the exhaustive list of contact_ids
        df_sub = pd.read_csv(self.config.SAMPLE_SUBMISSION_PATH)

        # Left merge: Keep all rows from sample_submission.
        # Rows present in df_preds get the prediction.
        # Rows missing from df_preds (filtered by gating) get NaN.
        df_final = df_sub[["contact_id"]].merge(df_preds, on="contact_id", how="left")

        # 5. Fill Missing Values
        # Missing values imply the pair was outside the gating distance -> No Contact (0)
        df_final["contact"] = df_final["contact_pred"].fillna(0).astype(int)

        # 6. Save Submission
        out_path = self.config.SUBMISSION_PATH
        df_final[["contact_id", "contact"]].to_csv(out_path, index=False)
        print(f"Submission saved to {out_path}")

    def run(self, load_cached_data=True):
        """
        Executes the evaluation pipeline.
        """
        print("Starting Evaluation Pipeline...")

        # 1. Optimize Threshold
        best_threshold = self.optimize_threshold(load_cached_data=load_cached_data)

        # 2. Generate Submission
        self.generate_submission(best_threshold, load_cached_data=load_cached_data)

        print("Evaluation Pipeline Completed Successfully.")
