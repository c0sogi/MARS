import os
import json
import gc
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import setup_seed
from library.data_loader import DataLoader
from library.feature_engineering import FeatureGenerator
from library.model import ContactXGB


class InferencePipeline:
    """
    Orchestrates the inference process for the Ego-Centric Dual-Stream GBDT architecture.
    Generates predictions for the test set and creates the submission file.
    """

    def __init__(self):
        self.working_dir = Config.WORKING_DIR
        self.thresholds_path = os.path.join(self.working_dir, "thresholds.json")
        self.submission_path = Config.SUBMISSION_OUTPUT_PATH
        setup_seed(Config.SEED)

    def _load_thresholds(self):
        """
        Loads optimized thresholds from the training run.
        Defaults to 0.5 if file is missing (e.g., during dry runs without training).
        """
        if os.path.exists(self.thresholds_path):
            with open(self.thresholds_path, "r") as f:
                thresholds = json.load(f)
            print(f"Loaded thresholds: {thresholds}")
            return thresholds
        else:
            print(
                f"Warning: Thresholds file not found at {self.thresholds_path}. Using default 0.5."
            )
            return {"stream_a": 0.5, "stream_b": 0.5}

    def _predict_stream(self, stream_name, df_merged, threshold):
        """
        Generates features and predictions for a specific stream.

        Args:
            stream_name (str): 'stream_a' or 'stream_b'.
            df_merged (pd.DataFrame): Merged test data.
            threshold (float): Decision threshold for this stream.

        Returns:
            pd.DataFrame: DataFrame containing 'contact_id' and 'contact' predictions.
        """
        print(f"\nProcessing {stream_name}...")

        # 1. Generate Features
        # Note: We set load_cached_data=False for inference to ensure we process the exact test set provided
        fg = FeatureGenerator(run_mode="test")
        X_test, _, ids_test = fg.generate_features(
            df_merged, stream=stream_name, load_cached_data=False
        )

        if len(X_test) == 0:
            print(f"No samples found for {stream_name}.")
            return pd.DataFrame(columns=["contact_id", "contact"])

        # 2. Load Model
        # We initialize with the config params to set up the wrapper, then load weights
        if stream_name == "stream_a":
            params = Config.XGB_PARAMS_STREAM_A
        else:
            params = Config.XGB_PARAMS_STREAM_B

        model = ContactXGB(params)
        model_path = os.path.join(self.working_dir, f"model_{stream_name}.json")

        try:
            model.load(model_path)
        except FileNotFoundError:
            print(
                f"Error: Model file for {stream_name} not found at {model_path}. Returning zeros."
            )
            return pd.DataFrame({"contact_id": ids_test, "contact": 0})

        # 3. Predict Probabilities
        print(f"Predicting {len(X_test)} samples for {stream_name}...")
        y_pred_proba = model.predict_proba(X_test)

        # 4. Apply Threshold
        y_pred_binary = (y_pred_proba >= threshold).astype(int)

        # 5. Format Result
        result_df = pd.DataFrame({"contact_id": ids_test, "contact": y_pred_binary})

        # Cleanup
        del X_test, ids_test, y_pred_proba, y_pred_binary, model
        gc.collect()

        return result_df

    def run_inference(self):
        """
        Main execution method for the inference pipeline.
        """
        print("Initializing Inference Pipeline...")

        # 1. Load Thresholds
        thresholds = self._load_thresholds()

        # 2. Load and Merge Test Data
        print("\n--- Loading Test Data ---")
        loader = DataLoader(run_mode="test")

        # Load metadata (derived from sample_submission)
        meta_test = loader.load_metadata()

        # Load tracking and helmets
        # Note: For test set, we load all available data associated with the plays in metadata
        unique_plays = meta_test["game_play"].unique()
        track_test = loader.load_tracking(unique_plays)
        helm_test = loader.load_helmets(unique_plays)

        # Merge
        # We assume inference is always a fresh run, so we can force reload or use cache if available.
        # Given the prompt requirement "without caching for test" in description, we set load_cached_data=False
        df_test_merged = loader.merge_data(
            meta_test, track_test, helm_test, load_cached_data=False
        )

        # Clear raw data
        del meta_test, track_test, helm_test
        gc.collect()

        # 3. Predict Stream A (Interaction)
        preds_a = self._predict_stream(
            "stream_a", df_test_merged, thresholds.get("stream_a", 0.5)
        )

        # 4. Predict Stream B (Impact)
        preds_b = self._predict_stream(
            "stream_b", df_test_merged, thresholds.get("stream_b", 0.5)
        )

        # 5. Combine and Save
        print("\n--- Generating Submission ---")

        # Concatenate predictions from both streams
        all_preds = pd.concat([preds_a, preds_b], axis=0)

        # Load sample submission to ensure correct order and completeness
        sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

        # Merge predictions into sample submission structure
        # We use left join on sample_sub to ensure we have exactly the rows required
        submission = sample_sub[["contact_id"]].merge(
            all_preds, on="contact_id", how="left"
        )

        # Fill missing values with 0 (safe default)
        submission["contact"] = submission["contact"].fillna(0).astype(int)

        # Save
        os.makedirs(os.path.dirname(self.submission_path), exist_ok=True)
        submission.to_csv(self.submission_path, index=False)

        print(f"Submission saved to {self.submission_path}")
        print(f"Submission shape: {submission.shape}")
        print("Inference Pipeline Completed Successfully.")
