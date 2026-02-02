import pandas as pd
import numpy as np
import os
from sklearn.model_selection import GroupKFold
from library.config import Config
from library.utils import (
    set_seed,
    optimize_threshold,
    save_to_cache,
    load_from_cache,
    get_cache_path,
    compute_mcc,
)
from library.data_manager import DataManager
from library.feature_builder import FeatureBuilder
from library.model_handler import XGBWrapper


class WorkflowManager:
    """
    Orchestrates the Sequential Cascade Dual-Stream GBDT pipeline.
    Manages data loading, feature engineering, OOF generation, model training, and inference.
    """

    def __init__(self):
        self.data_manager = DataManager()
        self.feature_builder = FeatureBuilder()
        self.working_dir = Config.WORKING_DIR
        set_seed(Config.SEED)

    def train_interaction_stream(self, load_cached_data=True):
        """
        Trains the Interaction Stream (Stream A).

        Returns:
            tuple: (final_model, best_threshold, val_preds_df)
        """
        print("\n=== Starting Stream A (Interaction) Training ===")

        # 1. Load Data
        df_train = self.data_manager.load_data("train", load_cached_data)
        df_val = self.data_manager.load_data("validation", load_cached_data)

        # 2. Build Features
        X_train, y_train, ids_train = self.feature_builder.build_stream_a_features(
            df_train, load_cached_data, split="train"
        )
        X_val, y_val, ids_val = self.feature_builder.build_stream_a_features(
            df_val, load_cached_data, split="validation"
        )

        # 3. Train Final Model on Full Training Set
        print("Training Final Stream A Model on full training set...")
        model_path = os.path.join(self.working_dir, "model_stream_a.json")
        final_model = XGBWrapper(Config.get_xgb_params("A"), model_path)

        # Fit on Train, Validate on Val
        final_model.fit(X_train, y_train, X_val, y_val)

        # 4. Predict on Validation Set and Optimize Threshold
        print("Optimizing Stream A Threshold...")
        val_probs = final_model.predict(X_val)
        best_thresh, best_score = optimize_threshold(y_val, val_probs)

        print(f"Stream A - Best Threshold: {best_thresh:.4f}")
        print(f"Stream A - Best Val MCC: {best_score}")  # Full precision

        val_preds_df = pd.DataFrame({"contact_id": ids_val, "prob": val_probs})

        return final_model, best_thresh, val_preds_df

    def train_impact_stream(self, load_cached_data=True):
        """
        Trains the Impact Stream (Stream B) using explicit kinematics (no cascade).

        Returns:
            tuple: (model, best_threshold)
        """
        print("\n=== Starting Stream B (Impact) Training ===")

        # 1. Load Data
        df_train = self.data_manager.load_data("train", load_cached_data)
        df_val = self.data_manager.load_data("validation", load_cached_data)

        # 2. Build Features (Independent)
        X_train, y_train, ids_train = self.feature_builder.build_stream_b_features(
            df_train, load_cached_data, split="train"
        )
        X_val, y_val, ids_val = self.feature_builder.build_stream_b_features(
            df_val, load_cached_data, split="validation"
        )

        # 3. Train Model
        print("Training Stream B Model...")
        model_path = os.path.join(self.working_dir, "model_stream_b.json")
        model = XGBWrapper(Config.get_xgb_params("B"), model_path)

        model.fit(X_train, y_train, X_val, y_val)

        # 4. Optimize Threshold
        print("Optimizing Stream B Threshold...")
        val_probs = model.predict(X_val)
        best_thresh, best_score = optimize_threshold(y_val, val_probs)

        print(f"Stream B - Best Threshold: {best_thresh:.4f}")
        print(f"Stream B - Best Val MCC: {best_score}")  # Full precision

        return model, best_thresh

    def run_inference_cascade(
        self, model_a, thresh_a, model_b, thresh_b, load_cached_data=True
    ):
        """
        Runs the independent inference on the Test set.

        Args:
            model_a: Trained Stream A model.
            thresh_a: Optimized threshold for Stream A.
            model_b: Trained Stream B model.
            thresh_b: Optimized threshold for Stream B.
        """
        print("\n=== Starting Inference on Test Set ===")

        # 1. Load Test Data
        df_test = self.data_manager.load_data("test", load_cached_data)

        # 2. Stream A Inference
        print("Running Stream A Inference...")
        X_test_a, _, ids_test_a = self.feature_builder.build_stream_a_features(
            df_test, load_cached_data, split="test"
        )

        probs_a = model_a.predict(X_test_a)
        preds_a_binary = (probs_a >= thresh_a).astype(int)

        # Create DataFrame for Stream A predictions
        test_preds_a_df = pd.DataFrame(
            {"contact_id": ids_test_a, "contact": preds_a_binary}
        )

        # 3. Stream B Inference (Independent)
        print("Running Stream B Inference...")
        X_test_b, _, ids_test_b = self.feature_builder.build_stream_b_features(
            df_test, load_cached_data, split="test"
        )

        probs_b = model_b.predict(X_test_b)
        preds_b_binary = (probs_b >= thresh_b).astype(int)

        test_preds_b_df = pd.DataFrame(
            {"contact_id": ids_test_b, "contact": preds_b_binary}
        )

        # 4. Combine Predictions
        print("Combining predictions...")
        # Stream A handles p2 != 'G', Stream B handles p2 == 'G'
        # The feature builder splits ensured disjoint sets of contact_ids

        # We only need contact_id and contact for submission
        sub_a = test_preds_a_df[["contact_id", "contact"]]
        sub_b = test_preds_b_df[["contact_id", "contact"]]

        combined_preds = pd.concat([sub_a, sub_b], axis=0)

        # 5. Format Submission
        print("Formatting submission...")
        sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

        # Merge to ensure order and completeness
        # Left join on sample submission to keep its order
        final_sub = pd.merge(
            sample_sub[["contact_id"]], combined_preds, on="contact_id", how="left"
        )

        # Fill missing (should not happen if logic is correct, but safe fallback is 0)
        missing_count = final_sub["contact"].isna().sum()
        if missing_count > 0:
            print(
                f"Warning: {missing_count} contact_ids missing from predictions. Filling with 0."
            )
            final_sub["contact"] = final_sub["contact"].fillna(0)

        final_sub["contact"] = final_sub["contact"].astype(int)

        # Save
        print(f"Saving submission to {Config.SUBMISSION_PATH}")
        final_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print("Submission saved successfully.")
