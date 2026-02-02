import os
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import seed_everything
from library.feature_engine import FeatureEngine
from library.model_zoo import LGBMExpert, XGBExpert, HistGBExpert


class InferencePipeline:
    """
    Manages the inference phase for the DEIB-AME solution.
    Loads the trained Tri-Model Expert Ensemble and the optimized decision threshold,
    generates features for the test set, and produces the final submission CSV.
    """

    def __init__(self):
        self.feature_engine = FeatureEngine()
        self.models_dir = os.path.join(Config.WORKING_DIR, "models")
        self.expert_dir = os.path.join(self.models_dir, "experts")
        self.threshold_path = os.path.join(self.models_dir, "best_threshold.npy")

        # Columns to exclude from inference features (Metadata)
        self.ignore_cols = [
            "contact",
            "contact_id",
            "game_play",
            "step",
            "nfl_player_id_1",
            "nfl_player_id_2",
        ]

    def _load_experts(self):
        """
        Instantiates and loads the three expert models from the ensemble.
        """
        experts = {}
        model_types = ["lgbm", "xgb", "hgb"]

        print("Loading Expert Ensemble...")

        for model_type in model_types:
            path = os.path.join(self.expert_dir, f"{model_type}_model.joblib")

            if model_type == "lgbm":
                model = LGBMExpert()
            elif model_type == "xgb":
                model = XGBExpert()
            elif model_type == "hgb":
                model = HistGBExpert()
            else:
                continue

            try:
                model.load(path)
                experts[model_type] = model
            except FileNotFoundError:
                print(
                    f"Warning: Model file not found at {path}. Skipping {model_type}."
                )

        if not experts:
            raise RuntimeError(
                "No expert models could be loaded. Ensure training has completed."
            )

        return experts

    def _load_threshold(self):
        """
        Loads the optimized decision threshold from disk.
        Defaults to 0.5 if not found.
        """
        if os.path.exists(self.threshold_path):
            threshold = np.load(self.threshold_path)[0]
            print(f"Loaded optimized threshold: {threshold}")
            return threshold
        else:
            print("Warning: Optimized threshold file not found. Defaulting to 0.5.")
            return 0.5

    def generate_submission(self, load_cached_data=True, sample_size=None):
        """
        Executes the full inference pipeline.

        Args:
            load_cached_data (bool): Whether to use cached feature files.
            sample_size (int, optional): Number of rows to process for debugging.
        """
        seed_everything(Config.SEED)

        # 1. Load Resources
        experts = self._load_experts()
        threshold = self._load_threshold()

        # 2. Process Test Data
        print("Processing test data features...")
        df_test = self.feature_engine.process_test(
            load_cached_data=load_cached_data, sample_size=sample_size
        )

        # 3. Prepare Features
        # Drop metadata columns to match training feature set
        X_test = df_test.drop(columns=self.ignore_cols, errors="ignore")

        # 4. Ensemble Inference
        print(
            f"Running inference on {len(X_test)} samples with {len(experts)} models..."
        )

        # Initialize probabilities with zeros
        probs_sum = np.zeros(len(X_test))

        for name, model in experts.items():
            probs = model.predict_proba(X_test)
            probs_sum += probs

        # Average probabilities
        y_prob = probs_sum / len(experts)

        # Apply Threshold
        y_pred = (y_prob >= threshold).astype(int)

        # 5. Format Submission
        # The feature engine applies gating, so df_test is a subset of the full submission.
        # We must merge predictions back into the full sample_submission template.
        print("Formatting final submission...")

        sample_sub_path = os.path.join(Config.INPUT_DIR, "sample_submission.csv")
        if not os.path.exists(sample_sub_path):
            raise FileNotFoundError(f"Sample submission not found at {sample_sub_path}")

        sample_sub = pd.read_csv(sample_sub_path)

        # Create a dataframe of predictions for the survivors
        pred_df = pd.DataFrame(
            {"contact_id": df_test["contact_id"], "contact_pred": y_pred}
        )

        # Merge: Left join ensures we have all rows from sample_submission.
        # Rows filtered out by gating (quadratic reachability) are assumed 0 (No Contact).
        submission = sample_sub.drop(columns=["contact"], errors="ignore").merge(
            pred_df, on="contact_id", how="left"
        )

        # Fill missing predictions (gated out) with 0
        submission["contact"] = submission["contact_pred"].fillna(0).astype(int)
        submission = submission.drop(columns=["contact_pred"])

        # 6. Save
        submission.to_csv(Config.SUBMISSION_PATH, index=False)

        print(f"Submission saved to {Config.SUBMISSION_PATH}")
        print(f"Total Rows: {len(submission)}")
        print(f"Predicted Contacts: {submission['contact'].sum()}")
