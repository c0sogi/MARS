import pandas as pd
import numpy as np
import os
import xgboost as xgb
from library.config import Config
from library.data_loader import DataLoader
from library.feature_engineering import FeatureEngine
from library.model_factory import ModelFactory
from library.utils import setup_seed, compute_mcc, CacheManager


class Trainer:
    """
    Orchestrates the training of the Dual-Stream architecture.
    Handles data loading, feature generation, undersampling, model training,
    and threshold optimization.
    """

    def __init__(self, debug=Config.DEBUG):
        self.debug = debug
        self.dl = DataLoader(debug=debug)
        self.fe = FeatureEngine(debug=debug)
        self.cache_manager = CacheManager()
        setup_seed(Config.SEED)

    def _get_feature_columns(self, stream):
        """
        Reconstructs the list of feature columns (including lags) for a given stream.
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

    def _undersample(self, df):
        """
        Applies targeted majority undersampling.
        Retains 100% of positives and subsamples negatives to Config.NEG_POS_RATIO.
        """
        pos = df[df["contact"] == 1]
        neg = df[df["contact"] == 0]

        n_pos = len(pos)
        n_neg = int(n_pos * Config.NEG_POS_RATIO)

        # If we have fewer negatives than the target ratio, keep all negatives
        if len(neg) > n_neg:
            neg = neg.sample(n=n_neg, random_state=Config.SEED)

        df_sampled = (
            pd.concat([pos, neg], axis=0)
            .sample(frac=1, random_state=Config.SEED)
            .reset_index(drop=True)
        )
        return df_sampled

    def _prepare_stream_data(
        self, stream_name, df_meta, df_tracking, df_helmets, is_train=True
    ):
        """
        Generates features, filters columns, and optionally undersamples data.
        Implements caching for the final processed dataframe.
        """
        # 1. Define Cache ID
        # We include the metadata length and head as a signature
        meta_sig = f"{len(df_meta)}_{df_meta['game_play'].iloc[0] if not df_meta.empty else 'empty'}"
        config_dict = {
            "function": "prepare_stream_data",
            "stream": stream_name,
            "is_train": is_train,
            "meta_sig": meta_sig,
            "debug": self.debug,
            "neg_pos_ratio": Config.NEG_POS_RATIO if is_train else "None",
        }
        cache_id = self.cache_manager.generate_cache_id(
            config_dict,
            prefix=(
                f"train_data_{stream_name}" if is_train else f"val_data_{stream_name}"
            ),
        )

        # 2. Try Load
        cached_data = self.cache_manager.load(cache_id, file_type="parquet")
        if cached_data is not None:
            return cached_data

        # 3. Process
        # Filter metadata for the specific stream
        if stream_name == "A":
            # Player-Player: P2 is not Ground
            df_stream = df_meta[df_meta["nfl_player_id_2"] != "G"].copy()
            if df_stream.empty:
                return pd.DataFrame()

            # Generate Features
            df_features = self.fe.generate_stream_a_features(
                df_stream, df_tracking, df_helmets
            )
        else:
            # Player-Ground: P2 is Ground
            df_stream = df_meta[df_meta["nfl_player_id_2"] == "G"].copy()
            if df_stream.empty:
                return pd.DataFrame()

            # Generate Features
            df_features = self.fe.generate_stream_b_features(df_stream, df_tracking)

        # Undersample if training
        if is_train:
            df_features = self._undersample(df_features)

        # 4. Save
        self.cache_manager.save(df_features, cache_id, file_type="parquet")

        return df_features

    def optimize_threshold(self, y_true, y_proba):
        """
        Finds the probability threshold that maximizes MCC.
        """
        best_threshold = 0.5
        best_score = -1.0

        thresholds = np.arange(
            Config.THRESHOLD_START, Config.THRESHOLD_END, Config.THRESHOLD_STEP
        )

        for thresh in thresholds:
            y_pred = (y_proba >= thresh).astype(int)
            score = compute_mcc(y_true, y_pred)

            if score > best_score:
                best_score = score
                best_threshold = thresh

        return best_threshold, best_score

    def train(self):
        """
        Main training loop for both streams.
        """
        print("Loading raw data...")
        # Load Metadata
        df_train_meta = self.dl.load_metadata("train")
        df_val_meta = self.dl.load_metadata("validation")

        # Load Inputs
        # Validation uses training tracking/helmets
        df_tracking = self.dl.load_tracking("train")
        df_helmets = self.dl.load_helmets("train")

        results = {}

        # =========================================================================
        # Stream A: The Collider (Player-Player)
        # =========================================================================
        print("\n=== Training Stream A: The Collider (Player-Player) ===")

        # Prepare Data
        df_train_A = self._prepare_stream_data(
            "A", df_train_meta, df_tracking, df_helmets, is_train=True
        )
        df_val_A = self._prepare_stream_data(
            "A", df_val_meta, df_tracking, df_helmets, is_train=False
        )

        if not df_train_A.empty and not df_val_A.empty:
            feature_cols_A = self._get_feature_columns("A")
            # Ensure columns exist (handle potential missing lags if window is small)
            feature_cols_A = [c for c in feature_cols_A if c in df_train_A.columns]

            X_train_A = df_train_A[feature_cols_A]
            y_train_A = df_train_A["contact"]
            X_val_A = df_val_A[feature_cols_A]
            y_val_A = df_val_A["contact"]

            # Train Model
            clf_A = ModelFactory.get_classifier("A")
            clf_A.fit(
                X_train_A,
                y_train_A,
                eval_set=[(X_val_A, y_val_A)],
                early_stopping_rounds=Config.EARLY_STOPPING_ROUNDS,
                verbose=Config.VERBOSE_EVAL,
            )

            # Evaluate
            y_proba_A = clf_A.predict_proba(X_val_A)[:, 1]
            best_thresh_A, best_mcc_A = self.optimize_threshold(y_val_A, y_proba_A)

            print(f"Stream A Best Threshold: {best_thresh_A}")
            print(f"Stream A Validation MCC: {best_mcc_A}")

            results["A"] = {
                "model": clf_A,
                "threshold": best_thresh_A,
                "score": best_mcc_A,
                "features": feature_cols_A,
            }
        else:
            print("Warning: Stream A data empty.")

        # =========================================================================
        # Stream B: The Accelerometer (Player-Ground)
        # =========================================================================
        print("\n=== Training Stream B: The Accelerometer (Player-Ground) ===")

        # Prepare Data
        df_train_B = self._prepare_stream_data(
            "B", df_train_meta, df_tracking, df_helmets, is_train=True
        )
        df_val_B = self._prepare_stream_data(
            "B", df_val_meta, df_tracking, df_helmets, is_train=False
        )

        if not df_train_B.empty and not df_val_B.empty:
            feature_cols_B = self._get_feature_columns("B")
            feature_cols_B = [c for c in feature_cols_B if c in df_train_B.columns]

            X_train_B = df_train_B[feature_cols_B]
            y_train_B = df_train_B["contact"]
            X_val_B = df_val_B[feature_cols_B]
            y_val_B = df_val_B["contact"]

            # Train Model
            clf_B = ModelFactory.get_classifier("B")
            clf_B.fit(
                X_train_B,
                y_train_B,
                eval_set=[(X_val_B, y_val_B)],
                early_stopping_rounds=Config.EARLY_STOPPING_ROUNDS,
                verbose=Config.VERBOSE_EVAL,
            )

            # Evaluate
            y_proba_B = clf_B.predict_proba(X_val_B)[:, 1]
            best_thresh_B, best_mcc_B = self.optimize_threshold(y_val_B, y_proba_B)

            print(f"Stream B Best Threshold: {best_thresh_B}")
            print(f"Stream B Validation MCC: {best_mcc_B}")

            results["B"] = {
                "model": clf_B,
                "threshold": best_thresh_B,
                "score": best_mcc_B,
                "features": feature_cols_B,
            }
        else:
            print("Warning: Stream B data empty.")

        return results
