import os
import json
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import seed_everything, optimize_threshold, compute_mcc
from library.features import FeatureEngineer
from library.model_factory import DualStreamGBDT


class Pipeline:
    def __init__(self):
        self.config = Config
        seed_everything(self.config.SEED)
        self.feature_engineer = FeatureEngineer()
        self.model = DualStreamGBDT()
        self.thresholds = {"stream_a": 0.5, "stream_b": 0.5}
        self.thresholds_path = os.path.join(self.config.WORKING_DIR, "thresholds.json")

    def run_training(self, load_cached_data: bool = True):
        """
        Executes the training pipeline:
        1. Loads/Computes features for Train and Validation.
        2. Trains the DualStreamGBDT model.
        3. Optimizes thresholds on Validation set.
        4. Saves models and thresholds.
        """
        print("=== Starting Training Pipeline ===")

        # 1. Get Data
        print("Retrieving Training Data...")
        data_train = self.feature_engineer.process_features(
            "train", load_cached_data=load_cached_data
        )

        print("Retrieving Validation Data...")
        data_val = self.feature_engineer.process_features(
            "validation", load_cached_data=load_cached_data
        )

        # 2. Train Models
        self.model.train(data_train, data_val)

        # 3. Optimize Thresholds
        print("\n=== Optimizing Thresholds ===")
        val_preds_proba = self.model.predict_proba(data_val)

        # Stream A Optimization
        _, y_val_a, ids_a = data_val["stream_a"]
        if len(y_val_a) > 0:
            thresh_a, mcc_a = optimize_threshold(y_val_a, val_preds_proba["stream_a"])
            self.thresholds["stream_a"] = thresh_a
            print(f"Stream A (Interaction) - Best Threshold: {thresh_a}, MCC: {mcc_a}")
        else:
            print("Stream A Validation set is empty. Keeping default threshold.")

        # Stream B Optimization
        _, y_val_b, ids_b = data_val["stream_b"]
        if len(y_val_b) > 0:
            thresh_b, mcc_b = optimize_threshold(y_val_b, val_preds_proba["stream_b"])
            self.thresholds["stream_b"] = thresh_b
            print(f"Stream B (Impact) - Best Threshold: {thresh_b}, MCC: {mcc_b}")
        else:
            print("Stream B Validation set is empty. Keeping default threshold.")

        # 4. Global Validation Metrics
        print("\n=== Global Validation Performance ===")
        # Combine predictions
        # Stream A
        preds_a = (val_preds_proba["stream_a"] >= self.thresholds["stream_a"]).astype(
            int
        )
        df_a = pd.DataFrame(
            {"contact_id": ids_a, "contact_pred": preds_a, "contact_true": y_val_a}
        )

        # Stream B
        preds_b = (val_preds_proba["stream_b"] >= self.thresholds["stream_b"]).astype(
            int
        )
        df_b = pd.DataFrame(
            {"contact_id": ids_b, "contact_pred": preds_b, "contact_true": y_val_b}
        )

        # Concat
        df_val_all = pd.concat([df_a, df_b], ignore_index=True)

        if not df_val_all.empty:
            overall_mcc = compute_mcc(
                df_val_all["contact_true"].values, df_val_all["contact_pred"].values
            )
            print(f"Overall Validation MCC: {overall_mcc}")
        else:
            print("Validation set is empty.")

        # 5. Save Artifacts
        self.model.save(self.config.WORKING_DIR)

        with open(self.thresholds_path, "w") as f:
            json.dump(self.thresholds, f)
        print(f"Thresholds saved to {self.thresholds_path}")

    def run_inference(self, load_cached_data: bool = True):
        """
        Executes the inference pipeline:
        1. Loads/Computes features for Test.
        2. Loads trained models (if not in memory).
        3. Predicts probabilities and applies optimized thresholds.
        4. Formats and saves submission.
        """
        print("\n=== Starting Inference Pipeline ===")

        # 1. Get Data
        print("Retrieving Test Data...")
        data_test = self.feature_engineer.process_features(
            "test", load_cached_data=load_cached_data
        )

        # 2. Load Thresholds (if available)
        if os.path.exists(self.thresholds_path):
            with open(self.thresholds_path, "r") as f:
                self.thresholds = json.load(f)
            print(f"Loaded thresholds: {self.thresholds}")
        else:
            print(
                f"Warning: Thresholds file not found. Using defaults: {self.thresholds}"
            )

        # Ensure model is loaded (if run separately, model might be None)
        if self.model.model_a is None and self.model.model_b is None:
            print("Loading models from disk...")
            self.model.load(self.config.WORKING_DIR)

        # 3. Predict
        print("Generating predictions...")
        preds_proba = self.model.predict_proba(data_test)

        # 4. Apply Thresholds & Format
        # Stream A
        _, _, ids_a = data_test["stream_a"]
        if len(ids_a) > 0:
            preds_a = (preds_proba["stream_a"] >= self.thresholds["stream_a"]).astype(
                int
            )
            df_a = pd.DataFrame({"contact_id": ids_a, "contact": preds_a})
        else:
            df_a = pd.DataFrame(columns=["contact_id", "contact"])

        # Stream B
        _, _, ids_b = data_test["stream_b"]
        if len(ids_b) > 0:
            preds_b = (preds_proba["stream_b"] >= self.thresholds["stream_b"]).astype(
                int
            )
            df_b = pd.DataFrame({"contact_id": ids_b, "contact": preds_b})
        else:
            df_b = pd.DataFrame(columns=["contact_id", "contact"])

        # Combine
        df_preds = pd.concat([df_a, df_b], ignore_index=True)

        # 5. Align with Sample Submission
        print("Formatting submission...")
        if not os.path.exists(self.config.SAMPLE_SUBMISSION_PATH):
            raise FileNotFoundError(
                f"Sample submission not found at {self.config.SAMPLE_SUBMISSION_PATH}"
            )

        df_sample = pd.read_csv(self.config.SAMPLE_SUBMISSION_PATH)

        # Merge predictions onto sample submission to ensure correct order and completeness
        # We drop the 'contact' column from sample first if it exists (it's usually all 0s)
        if "contact" in df_sample.columns:
            df_sample = df_sample.drop(columns=["contact"])

        final_submission = df_sample.merge(df_preds, on="contact_id", how="left")

        # Fill missing values with 0 (no contact)
        # This handles cases where features might have been filtered out or missing
        missing_count = final_submission["contact"].isnull().sum()
        if missing_count > 0:
            print(
                f"Warning: {missing_count} contact_ids missing from predictions. Filling with 0."
            )
            final_submission["contact"] = final_submission["contact"].fillna(0)

        final_submission["contact"] = final_submission["contact"].astype(int)

        # Save
        final_submission.to_csv(self.config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {self.config.SUBMISSION_PATH}")
