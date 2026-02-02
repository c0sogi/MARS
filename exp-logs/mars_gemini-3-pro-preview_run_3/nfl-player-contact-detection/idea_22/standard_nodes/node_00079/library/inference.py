import pandas as pd
import numpy as np
import os
from library.config import Config
from library.data_loader import DataLoader
from library.feature_engineering import FeatureEngine
from library.utils import CacheManager


class Predictor:
    """
    Manages the inference pipeline for the Dual-Stream architecture.
    Routes test data to the appropriate stream (A: Collider, B: Accelerometer),
    generates features, applies optimized thresholds, and formats the submission.
    """

    def __init__(self, model_a, thresh_a, model_b, thresh_b, debug=Config.DEBUG):
        """
        Args:
            model_a: Trained XGBClassifier for Stream A (Player-Player).
            thresh_a: Optimized probability threshold for Stream A.
            model_b: Trained XGBClassifier for Stream B (Player-Ground).
            thresh_b: Optimized probability threshold for Stream B.
            debug: Boolean flag for debug mode.
        """
        self.model_a = model_a
        self.thresh_a = thresh_a
        self.model_b = model_b
        self.thresh_b = thresh_b
        self.debug = debug

        self.dl = DataLoader(debug=debug)
        self.fe = FeatureEngine(debug=debug)
        self.cache_manager = CacheManager()

    def _get_feature_columns(self, stream):
        """
        Reconstructs the list of feature columns (including lags) for a given stream.
        Must match the order used during training.
        """
        if stream == "A":
            base_features = Config.STREAM_A_FEATURES
        elif stream == "B":
            base_features = Config.STREAM_B_FEATURES
        else:
            raise ValueError("Invalid stream")

        feature_cols = []
        for f in base_features:
            feature_cols.append(f)
            for lag in Config.LAG_OFFSETS:
                if lag != 0:
                    feature_cols.append(f"{f}_lag_{lag}")
        return feature_cols

    def _prepare_test_data(
        self, stream_name, df_meta, df_tracking, df_helmets, load_cached_data=True
    ):
        """
        Generates features for the test set with caching.
        """
        # 1. Define Cache ID
        # Signature based on metadata length and first element to distinguish from train/val
        meta_sig = f"{len(df_meta)}_{df_meta['game_play'].iloc[0] if not df_meta.empty else 'empty'}"
        config_dict = {
            "function": "prepare_test_data",
            "stream": stream_name,
            "mode": "test",
            "meta_sig": meta_sig,
            "debug": self.debug,
        }

        cache_id = self.cache_manager.generate_cache_id(
            config_dict, prefix=f"test_data_{stream_name}"
        )

        # 2. Try Load
        if load_cached_data:
            cached_data = self.cache_manager.load(cache_id, file_type="parquet")
            if cached_data is not None:
                return cached_data

        # 3. Process
        if stream_name == "A":
            # Player-Player
            df_features = self.fe.generate_stream_a_features(
                df_meta, df_tracking, df_helmets, load_cached_data=load_cached_data
            )
        else:
            # Player-Ground
            df_features = self.fe.generate_stream_b_features(
                df_meta, df_tracking, load_cached_data=load_cached_data
            )

        # 4. Save
        self.cache_manager.save(df_features, cache_id, file_type="parquet")

        return df_features

    def predict(self):
        """
        Executes the full prediction pipeline and saves the submission file.
        """
        print("Starting Inference...")

        # 1. Load Raw Test Data
        df_test_meta = self.dl.load_metadata("test")
        df_tracking = self.dl.load_tracking("test")
        df_helmets = self.dl.load_helmets("test")

        predictions = []

        # =========================================================================
        # Stream A: The Collider (Player-Player)
        # =========================================================================
        # Filter for Player-Player contacts
        df_test_A = df_test_meta[df_test_meta["nfl_player_id_2"] != "G"].copy()

        if not df_test_A.empty:
            print(f"Processing Stream A (Player-Player): {len(df_test_A)} samples")

            # Generate Features
            df_features_A = self._prepare_test_data(
                "A", df_test_A, df_tracking, df_helmets
            )

            # Get Feature Columns
            feature_cols_A = self._get_feature_columns("A")
            # Ensure columns exist (handle potential missing lags if any)
            # In inference, we must be strict or fill missing with 0
            missing_cols = set(feature_cols_A) - set(df_features_A.columns)
            if missing_cols:
                for c in missing_cols:
                    df_features_A[c] = 0

            X_test_A = df_features_A[feature_cols_A]

            # Predict
            y_proba_A = self.model_a.predict_proba(X_test_A)[:, 1]
            y_pred_A = (y_proba_A >= self.thresh_a).astype(int)

            # Store results
            result_A = df_features_A[["contact_id"]].copy()
            result_A["contact"] = y_pred_A
            predictions.append(result_A)

        # =========================================================================
        # Stream B: The Accelerometer (Player-Ground)
        # =========================================================================
        # Filter for Player-Ground contacts
        df_test_B = df_test_meta[df_test_meta["nfl_player_id_2"] == "G"].copy()

        if not df_test_B.empty:
            print(f"Processing Stream B (Player-Ground): {len(df_test_B)} samples")

            # Generate Features
            df_features_B = self._prepare_test_data(
                "B", df_test_B, df_tracking, df_helmets
            )

            # Get Feature Columns
            feature_cols_B = self._get_feature_columns("B")
            missing_cols = set(feature_cols_B) - set(df_features_B.columns)
            if missing_cols:
                for c in missing_cols:
                    df_features_B[c] = 0

            X_test_B = df_features_B[feature_cols_B]

            # Predict
            y_proba_B = self.model_b.predict_proba(X_test_B)[:, 1]
            y_pred_B = (y_proba_B >= self.thresh_b).astype(int)

            # Store results
            result_B = df_features_B[["contact_id"]].copy()
            result_B["contact"] = y_pred_B
            predictions.append(result_B)

        # =========================================================================
        # Consolidation & Submission
        # =========================================================================
        if predictions:
            df_pred = pd.concat(predictions, axis=0)
        else:
            # Fallback if empty (should not happen in valid test set)
            df_pred = pd.DataFrame(columns=["contact_id", "contact"])

        # Ensure we cover all IDs in the original test set and maintain order if possible
        # We merge back to the original metadata to ensure completeness
        df_submission = df_test_meta[["contact_id"]].merge(
            df_pred, on="contact_id", how="left"
        )

        # Fill missing predictions with 0 (default no contact)
        df_submission["contact"] = df_submission["contact"].fillna(0).astype(int)

        # Save
        save_path = Config.SUBMISSION_PATH
        df_submission.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")
        print(f"Total Predictions: {len(df_submission)}")
        print(f"Positive Predictions: {df_submission['contact'].sum()}")
