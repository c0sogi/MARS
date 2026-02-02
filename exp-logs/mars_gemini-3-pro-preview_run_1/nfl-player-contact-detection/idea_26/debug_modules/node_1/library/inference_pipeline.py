import pandas as pd
import numpy as np
import os
import gc
import library.config as config
from library.physics_engine import FeatureManager
from library.model_factory import TriEnsemble


class InferencePipeline:
    """
    Manages the inference process for the VASM-E strategy.
    Generates features for the test set, loads the expert ensemble,
    and produces the final submission file.
    """

    def __init__(self):
        self.feature_manager = FeatureManager()

    def run(self, load_cached_data=True, debug_sample=None):
        """
        Executes the inference pipeline.

        Args:
            load_cached_data (bool): Whether to use cached features if available.
            debug_sample (int, optional): Number of rows to sample for debugging.

        Returns:
            str: Path to the generated submission file.
        """
        print(f"Starting Inference Pipeline (Exp: {config.EXP_NAME})")

        # 1. Process Test Data
        # This returns only the rows that survived kinematic gating
        df_test_features = self.feature_manager.process_data(
            split="test", load_cached_data=load_cached_data, debug_sample=debug_sample
        )

        # 2. Load Model and Threshold
        model_path = "expert_ensemble.joblib"
        print(f"Loading model from {model_path}...")
        model = TriEnsemble.load(model_path)

        threshold_path = os.path.join(config.MODEL_DIR, "best_threshold.npy")
        if os.path.exists(threshold_path):
            best_threshold = float(np.load(threshold_path))
            print(f"Loaded optimized threshold: {best_threshold:.4f}")
        else:
            best_threshold = 0.5
            print(f"Threshold file not found. Defaulting to {best_threshold}")

        # 3. Generate Predictions for Survivors
        print(f"Predicting on {len(df_test_features)} gated test samples...")

        if len(df_test_features) > 0:
            X_test = df_test_features[config.MODEL_FEATURES]
            probs = model.predict_proba(X_test)
        else:
            print(
                "Warning: No test samples survived gating. All predictions will be 0."
            )
            probs = np.array([])

        # Create a mapping from contact_id to predicted probability
        # We use a temporary dataframe for robust merging
        pred_mapping = pd.DataFrame(
            {"contact_id": df_test_features["contact_id"], "prob": probs}
        )

        # 4. Assemble Submission
        # Load the sample submission to ensure we have ALL contact_ids
        # (including those filtered out by gating)
        print("Loading sample submission template...")
        df_submission = pd.read_csv(config.SAMPLE_SUBMISSION_PATH)

        if debug_sample is not None:
            # If debugging, we might not have all IDs processed, but we should
            # still output the full submission file format, just with 0s for unprocessed.
            pass

        # Merge predictions into the template
        # Left join ensures we keep all required rows
        print("Merging predictions...")
        df_submission = df_submission.merge(pred_mapping, on="contact_id", how="left")

        # Fill NaNs:
        # NaNs occur for rows that were filtered out by kinematic gating (distance > threshold).
        # These are physically impossible contacts, so probability is 0.0.
        # Also handles any potential merge misses.
        df_submission["prob"] = df_submission["prob"].fillna(0.0)

        # Apply Threshold
        df_submission["contact"] = (df_submission["prob"] >= best_threshold).astype(int)

        # 5. Save Output
        # Ensure output directory exists
        os.makedirs(config.SUBMISSION_DIR, exist_ok=True)

        output_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")

        # Select only required columns
        out_cols = ["contact_id", "contact"]
        df_submission[out_cols].to_csv(output_path, index=False)

        print(f"Submission saved to {output_path}")
        print(
            f"Positive predictions: {df_submission['contact'].sum()} / {len(df_submission)}"
        )

        # Clean up
        del df_test_features, df_submission, pred_mapping, model
        gc.collect()

        return output_path
