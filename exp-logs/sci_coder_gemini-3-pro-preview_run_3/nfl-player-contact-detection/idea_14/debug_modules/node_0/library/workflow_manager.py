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

    def _generate_oof_stream_a(self, X, y, ids, groups, load_cached_data=True):
        """
        Generates Out-of-Fold predictions for Stream A to be used as context for Stream B.
        Uses GroupKFold to prevent leakage across plays.
        """
        cache_config = {
            "task": "oof_generation_stream_a",
            "n_splits": 5,
            "seed": Config.SEED,
            "debug": Config.DEBUG,
        }
        cache_path = get_cache_path(
            self.working_dir, "oof_preds_stream_a", cache_config, "parquet"
        )

        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached OOF predictions from {cache_path}")
            return pd.read_parquet(cache_path)

        print("Generating OOF predictions for Stream A (5-Fold GroupKFold)...")

        # Initialize storage for OOF predictions
        oof_preds = []

        # 5-Fold Split
        gkf = GroupKFold(n_splits=5)

        # We need to reconstruct a DataFrame to map predictions back to contact_ids
        # X is a DataFrame, y is numpy array, ids is numpy array

        fold = 1
        for train_idx, val_idx in gkf.split(X, y, groups=groups):
            print(f"  Processing Fold {fold}/5...")

            # Slice data
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y[train_idx]
            X_val_fold = X.iloc[val_idx]
            # y_val_fold = y[val_idx] # Not needed for prediction
            ids_val_fold = ids[val_idx]

            # Train Model
            model = XGBWrapper(Config.get_xgb_params("A"), model_path=None)
            model.fit(X_train_fold, y_train_fold)

            # Predict
            probs = model.predict(X_val_fold)

            # Store results
            fold_df = pd.DataFrame({"contact_id": ids_val_fold, "prob": probs})
            oof_preds.append(fold_df)

            fold += 1

        # Concatenate all folds
        oof_df = pd.concat(oof_preds, axis=0).reset_index(drop=True)

        # Save to cache
        print(f"Saving OOF predictions to {cache_path}")
        save_to_cache(oof_df, cache_path)

        return oof_df

    def train_interaction_stream(self, load_cached_data=True):
        """
        Trains the Interaction Stream (Stream A).

        Returns:
            tuple: (final_model, best_threshold, train_oof_df, val_preds_df)
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

        # 3. Generate OOF Predictions for Training Set (Context for Stream B)
        # We need groups for GroupKFold. The groups are 'game_play'.
        # We need to extract game_play corresponding to the rows in X_train.
        # Since X_train was built from df_train filtered for p2!='G', we need to replicate that filter to get groups.
        # However, feature_builder returns ids (contact_id). We can parse game_play from contact_id.

        # Vectorized parsing of game_play from contact_id array
        # contact_id format: game_key_play_id_step_p1_p2
        # We need the first two tokens joined by _.
        print("Extracting groups for OOF generation...")
        contact_ids_series = pd.Series(ids_train)
        # Split by '_' and take first two parts
        split_ids = contact_ids_series.str.split("_", n=2, expand=True)
        groups = split_ids[0] + "_" + split_ids[1]

        train_oof_df = self._generate_oof_stream_a(
            X_train, y_train, ids_train, groups, load_cached_data
        )

        # 4. Train Final Model on Full Training Set
        print("Training Final Stream A Model on full training set...")
        model_path = os.path.join(self.working_dir, "model_stream_a.json")
        final_model = XGBWrapper(Config.get_xgb_params("A"), model_path)

        # Fit on Train, Validate on Val
        final_model.fit(X_train, y_train, X_val, y_val)

        # 5. Predict on Validation Set and Optimize Threshold
        print("Optimizing Stream A Threshold...")
        val_probs = final_model.predict(X_val)
        best_thresh, best_score = optimize_threshold(y_val, val_probs)

        print(f"Stream A - Best Threshold: {best_thresh:.4f}")
        print(f"Stream A - Best Val MCC: {best_score}")  # Full precision

        val_preds_df = pd.DataFrame({"contact_id": ids_val, "prob": val_probs})

        return final_model, best_thresh, train_oof_df, val_preds_df

    def train_impact_stream(
        self, train_context_df, val_context_df, load_cached_data=True
    ):
        """
        Trains the Impact Stream (Stream B) using context from Stream A.

        Args:
            train_context_df (pd.DataFrame): Stream A predictions for training set (OOF).
            val_context_df (pd.DataFrame): Stream A predictions for validation set.

        Returns:
            tuple: (model, best_threshold)
        """
        print("\n=== Starting Stream B (Impact) Training ===")

        # 1. Load Data
        df_train = self.data_manager.load_data("train", load_cached_data)
        df_val = self.data_manager.load_data("validation", load_cached_data)

        # 2. Build Features (with Context)
        X_train, y_train, ids_train = self.feature_builder.build_stream_b_features(
            df_train, train_context_df, load_cached_data, split="train"
        )
        X_val, y_val, ids_val = self.feature_builder.build_stream_b_features(
            df_val, val_context_df, load_cached_data, split="validation"
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
        Runs the inference cascade on the Test set.

        Args:
            model_a: Trained Stream A model.
            thresh_a: Optimized threshold for Stream A.
            model_b: Trained Stream B model.
            thresh_b: Optimized threshold for Stream B.
        """
        print("\n=== Starting Inference Cascade on Test Set ===")

        # 1. Load Test Data
        df_test = self.data_manager.load_data("test", load_cached_data)

        # 2. Stream A Inference
        print("Running Stream A Inference...")
        X_test_a, _, ids_test_a = self.feature_builder.build_stream_a_features(
            df_test, load_cached_data, split="test"
        )

        probs_a = model_a.predict(X_test_a)
        preds_a_binary = (probs_a >= thresh_a).astype(int)

        # Create DataFrame for Stream A predictions (needed for context)
        test_preds_a_df = pd.DataFrame(
            {"contact_id": ids_test_a, "prob": probs_a, "contact": preds_a_binary}
        )

        # 3. Stream B Inference (using Stream A predictions as context)
        print("Running Stream B Inference...")
        X_test_b, _, ids_test_b = self.feature_builder.build_stream_b_features(
            df_test, test_preds_a_df, load_cached_data, split="test"
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
