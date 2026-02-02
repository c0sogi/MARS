import os
import json
import gc
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import optimize_threshold, calc_mcc
from library.feature_engineering import FeatureEngineer
from library.model_wrapper import DualStreamXGBoost


class PipelineManager:
    """
    Orchestrates the Invariant-Physics Temporal Pyramid Dual-Stream GBDT pipeline.
    Handles data preparation, model training, threshold optimization, and inference.
    """

    def __init__(self, debug=False):
        """
        Args:
            debug (bool): If True, runs in debug mode with smaller datasets.
        """
        self.debug = debug
        self.model_wrapper = DualStreamXGBoost()
        self.thresholds_path = os.path.join(Config.WORKING_DIR, "thresholds.json")

    def run_training_pipeline(self, load_cached_data=True):
        """
        Executes the training pipeline:
        1. Feature Engineering (Train & Validation)
        2. Model Training (Stream A & B)
        3. Threshold Optimization
        4. Model & Threshold Persistence

        Args:
            load_cached_data (bool): Whether to use cached features.

        Returns:
            dict: Optimized thresholds for Stream A and Stream B.
        """
        print("=== Starting Training Pipeline ===")

        # --- 1. Feature Engineering ---
        print("Initializing Feature Engineers...")
        fe_train = FeatureEngineer(mode="train", debug=self.debug)
        fe_val = FeatureEngineer(mode="validation", debug=self.debug)

        # Load Stream A Data (Player-Player)
        print("\n[Stream A] Preparing Data...")
        X_train_A, y_train_A, _ = fe_train.construct_stream_a(load_cached_data)
        X_val_A, y_val_A, _ = fe_val.construct_stream_a(load_cached_data)

        # Load Stream B Data (Player-Ground)
        print("\n[Stream B] Preparing Data...")
        X_train_B, y_train_B, _ = fe_train.construct_stream_b(load_cached_data)
        X_val_B, y_val_B, _ = fe_val.construct_stream_b(load_cached_data)

        # --- 2. Model Training ---
        print("\n[Stream A] Training Interaction Model...")
        if X_train_A.shape[0] > 0:
            self.model_wrapper.fit_stream(
                X_train_A, y_train_A, stream="A", X_val=X_val_A, y_val=y_val_A
            )
        else:
            print("Warning: No training data for Stream A.")

        print("\n[Stream B] Training Impact Model...")
        if X_train_B.shape[0] > 0:
            self.model_wrapper.fit_stream(
                X_train_B, y_train_B, stream="B", X_val=X_val_B, y_val=y_val_B
            )
        else:
            print("Warning: No training data for Stream B.")

        # Save Models
        self.model_wrapper.save_models()

        # --- 3. Threshold Optimization ---
        print("\n=== Optimizing Thresholds ===")
        thresholds = {"A": 0.5, "B": 0.5}

        # Optimize Stream A
        if self.model_wrapper.model_a is not None and X_val_A.shape[0] > 0:
            print("Predicting Stream A Validation Set...")
            probs_A = self.model_wrapper.predict_stream(X_val_A, stream="A")
            best_thresh_A, best_mcc_A = optimize_threshold(y_val_A, probs_A)
            thresholds["A"] = float(best_thresh_A)
            print(
                f"Stream A Optimal Threshold: {best_thresh_A:.4f}, Validation MCC: {best_mcc_A}"
            )

        # Optimize Stream B
        if self.model_wrapper.model_b is not None and X_val_B.shape[0] > 0:
            print("Predicting Stream B Validation Set...")
            probs_B = self.model_wrapper.predict_stream(X_val_B, stream="B")
            best_thresh_B, best_mcc_B = optimize_threshold(y_val_B, probs_B)
            thresholds["B"] = float(best_thresh_B)
            print(
                f"Stream B Optimal Threshold: {best_thresh_B:.4f}, Validation MCC: {best_mcc_B}"
            )

        # Save Thresholds
        with open(self.thresholds_path, "w") as f:
            json.dump(thresholds, f, indent=4)
        print(f"Thresholds saved to {self.thresholds_path}")

        # Clean up memory
        del X_train_A, y_train_A, X_val_A, y_val_A
        del X_train_B, y_train_B, X_val_B, y_val_B
        gc.collect()

        return thresholds

    def run_inference_pipeline(self, thresholds=None, load_cached_data=True):
        """
        Executes the inference pipeline:
        1. Feature Engineering (Test)
        2. Model Loading
        3. Prediction (Stream A & B)
        4. Submission Generation

        Args:
            thresholds (dict, optional): Thresholds to use. If None, loads from disk.
            load_cached_data (bool): Whether to use cached features.
        """
        print("=== Starting Inference Pipeline ===")

        # Load Thresholds if not provided
        if thresholds is None:
            if os.path.exists(self.thresholds_path):
                with open(self.thresholds_path, "r") as f:
                    thresholds = json.load(f)
                print(f"Loaded thresholds: {thresholds}")
            else:
                print("Warning: No thresholds found. Using default 0.5.")
                thresholds = {"A": 0.5, "B": 0.5}

        # Load Models
        self.model_wrapper.load_models()

        # --- 1. Feature Engineering (Test) ---
        print("Initializing Test Feature Engineer...")
        fe_test = FeatureEngineer(mode="test", debug=self.debug)

        # Stream A
        print("\n[Stream A] Preparing Test Data...")
        X_test_A, _, ids_test_A = fe_test.construct_stream_a(load_cached_data)

        # Stream B
        print("\n[Stream B] Preparing Test Data...")
        X_test_B, _, ids_test_B = fe_test.construct_stream_b(load_cached_data)

        # --- 2. Prediction ---
        results = []

        # Stream A Predictions
        if self.model_wrapper.model_a is not None and X_test_A.shape[0] > 0:
            print(f"Predicting {X_test_A.shape[0]} samples for Stream A...")
            probs_A = self.model_wrapper.predict_stream(X_test_A, stream="A")
            preds_A = (probs_A >= thresholds["A"]).astype(int)

            df_A = pd.DataFrame({"contact_id": ids_test_A, "contact": preds_A})
            results.append(df_A)

        # Stream B Predictions
        if self.model_wrapper.model_b is not None and X_test_B.shape[0] > 0:
            print(f"Predicting {X_test_B.shape[0]} samples for Stream B...")
            probs_B = self.model_wrapper.predict_stream(X_test_B, stream="B")
            preds_B = (probs_B >= thresholds["B"]).astype(int)

            df_B = pd.DataFrame({"contact_id": ids_test_B, "contact": preds_B})
            results.append(df_B)

        # --- 3. Submission Generation ---
        print("\nGenerating Submission File...")

        # Combine predictions
        if results:
            df_preds = pd.concat(results, ignore_index=True)
        else:
            df_preds = pd.DataFrame(columns=["contact_id", "contact"])

        # Load Sample Submission to ensure correct order and completeness
        df_sample = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

        # Merge predictions into sample submission
        # We use left join on sample submission to ensure we output exactly what is required
        df_submission = pd.merge(
            df_sample[["contact_id"]], df_preds, on="contact_id", how="left"
        )

        # Fill missing predictions with 0 (No Contact)
        # This handles cases where features might have been filtered out or missing
        missing_count = df_submission["contact"].isnull().sum()
        if missing_count > 0:
            print(
                f"Warning: {missing_count} contact_ids were not predicted. Filling with 0."
            )
            df_submission["contact"] = df_submission["contact"].fillna(0)

        # Ensure integer type
        df_submission["contact"] = df_submission["contact"].astype(int)

        # Save
        df_submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
        print(f"Submission Shape: {df_submission.shape}")

        # Clean up
        del X_test_A, X_test_B, df_preds, df_submission
        gc.collect()
