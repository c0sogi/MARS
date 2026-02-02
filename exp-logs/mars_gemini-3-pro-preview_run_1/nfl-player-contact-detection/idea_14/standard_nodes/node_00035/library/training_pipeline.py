import os
import numpy as np
import pandas as pd
import joblib
from typing import Tuple, List, Optional, Dict

from library.config import (
    ModelConfig,
    FeatureConfig,
    WORKING_DIR,
    MODEL_OUTPUT_DIR,
    CACHE_DIR,
    SEED,
)
from library.data_factory import DataFactory
from library.model_factory import LGBMWrapper, XGBWrapper, EnsemblePredictor
from library.utils import set_seed, compute_mcc, get_hashed_filepath


class TrainingPipeline:
    """
    Orchestrates the Mining Curriculum: Scout Training -> Hard Negative Mining -> Expert Training.
    """

    def __init__(self, debug: bool = False):
        self.debug = debug
        self.model_config = ModelConfig()
        self.feature_config = FeatureConfig()
        self.data_factory = DataFactory(self.feature_config)

        # Adjust config for debugging
        if self.debug:
            print("Debug mode enabled: Reducing estimators and data size.")
            self.model_config.lgbm_params["n_estimators"] = 50
            self.model_config.xgb_params["n_estimators"] = 50
            self.model_config.scout_n_estimators = 50

        set_seed(SEED)

    def _prepare_xy(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separates features and target from the dataframe.
        """
        drop_cols = [
            "contact_id",
            "game_play",
            "step",
            "nfl_player_id_1",
            "nfl_player_id_2",
            "contact",
        ]
        # Filter out non-numeric columns just in case, though feature engineering should handle it
        feature_cols = [c for c in df.columns if c not in drop_cols]

        X = df[feature_cols]
        y = df["contact"]
        return X, y

    def run_scout_training(
        self, df_train: pd.DataFrame, df_val: pd.DataFrame
    ) -> LGBMWrapper:
        """
        Phase 1: Train a Scout model on a balanced subset to learn the coarse boundary.
        """
        print("\n--- Phase 1: Scout Training ---")

        # 1. Create Balanced Scout Dataset
        # Positives
        df_pos = df_train[df_train["contact"] == 1]
        # Random Negatives (1:1 ratio)
        n_pos = len(df_pos)
        df_neg = df_train[df_train["contact"] == 0].sample(n=n_pos, random_state=SEED)

        df_scout = (
            pd.concat([df_pos, df_neg])
            .sample(frac=1, random_state=SEED)
            .reset_index(drop=True)
        )

        print(f"Scout Dataset: {len(df_scout)} rows (Balanced 1:1)")

        X_train, y_train = self._prepare_xy(df_scout)
        X_val, y_val = self._prepare_xy(df_val)

        # 2. Configure Scout Model
        # Use lighter settings for speed
        scout_params = self.model_config.lgbm_params.copy()
        scout_params["n_estimators"] = self.model_config.scout_n_estimators

        scout_model = LGBMWrapper(config=self.model_config, overrides=scout_params)

        # 3. Train
        scout_model.train(X_train, y_train, X_val, y_val)

        # 4. Save
        scout_model.save("scout_lgbm.joblib")

        return scout_model

    def mine_hard_negatives(
        self, df_train: pd.DataFrame, scout_model: LGBMWrapper
    ) -> pd.DataFrame:
        """
        Phase 2: Use Scout model to find Hard Negatives in the full gated training set.
        """
        print("\n--- Phase 2: Hard Negative Mining ---")

        # Check cache for hard negative indices
        cache_key = {
            "method": "scout_mining",
            "threshold": self.model_config.hard_negative_threshold,
            "data_len": len(df_train),
        }
        cache_path = get_hashed_filepath("hard_negative_indices", cache_key, "npy")

        df_hard_neg = None
        if os.path.exists(cache_path):
            print(f"Loading cached hard negative indices from {cache_path}...")
            try:
                hard_neg_indices = np.load(cache_path)
                # Cite debug_lesson_4: Verify indices match the current dataframe to prevent KeyError
                df_hard_neg = df_train.loc[hard_neg_indices]
            except KeyError:
                print(
                    "Cached indices mismatch with DataFrame index. Discarding cache and re-mining..."
                )
                df_hard_neg = None

        if df_hard_neg is None:
            print("Running inference on full training set to mine hard negatives...")
            X_full, y_full = self._prepare_xy(df_train)

            # Predict
            preds = scout_model.predict(X_full)

            # Identify Hard Negatives: Ground Truth = 0 AND Prediction > Threshold
            hard_neg_mask = (y_full == 0) & (
                preds > self.model_config.hard_negative_threshold
            )

            df_hard_neg = df_train[hard_neg_mask]
            hard_neg_indices = df_hard_neg.index.to_numpy()

            # Cache indices
            np.save(cache_path, hard_neg_indices)

        print(
            f"Mined {len(df_hard_neg)} Hard Negatives (Threshold > {self.model_config.hard_negative_threshold})"
        )
        return df_hard_neg

    def run_expert_training(
        self, df_train: pd.DataFrame, df_hard_neg: pd.DataFrame, df_val: pd.DataFrame
    ) -> Tuple[LGBMWrapper, XGBWrapper]:
        """
        Phase 3: Train Expert Ensemble on Positives + Hard Negatives + Random Buffer.
        """
        print("\n--- Phase 3: Expert Training ---")

        # 1. Construct Expert Dataset
        # All Positives
        df_pos = df_train[df_train["contact"] == 1]

        # Buffer of Random Negatives (equal to positives count to maintain diversity)
        # We exclude hard negatives from this sample to avoid duplication, though concat handles it.
        # Ideally, sample from (Negatives - Hard Negatives).
        # For simplicity and speed, sampling from all negatives is fine, duplicates are rare or handled.
        n_pos = len(df_pos)
        df_neg_buffer = df_train[df_train["contact"] == 0].sample(
            n=n_pos, random_state=SEED
        )

        # Combine: Positives + Hard Negatives + Random Buffer
        df_expert = pd.concat([df_pos, df_hard_neg, df_neg_buffer])

        # Drop duplicates if any (e.g. if a hard negative was also picked in random buffer)
        df_expert = df_expert.drop_duplicates(subset=["contact_id"])

        # Shuffle
        df_expert = df_expert.sample(frac=1, random_state=SEED).reset_index(drop=True)

        print(f"Expert Dataset: {len(df_expert)} rows")
        print(f"  - Positives: {len(df_pos)}")
        print(f"  - Hard Negatives: {len(df_hard_neg)}")
        print(f"  - Random Buffer: {len(df_neg_buffer)}")

        X_train, y_train = self._prepare_xy(df_expert)
        X_val, y_val = self._prepare_xy(df_val)

        # 2. Train Expert LightGBM
        print("\nTraining Expert LightGBM...")
        lgbm_expert = LGBMWrapper(config=self.model_config)
        lgbm_expert.train(X_train, y_train, X_val, y_val)
        lgbm_expert.save("expert_lgbm.joblib")

        # 3. Train Expert XGBoost
        print("\nTraining Expert XGBoost...")
        xgb_expert = XGBWrapper(config=self.model_config)
        xgb_expert.train(X_train, y_train, X_val, y_val)
        xgb_expert.save("expert_xgb.joblib")

        return lgbm_expert, xgb_expert

    def optimize_threshold(self, models: List[object], df_val: pd.DataFrame) -> float:
        """
        Optimizes the decision threshold on the validation set to maximize MCC.
        """
        print("\n--- Threshold Optimization ---")
        X_val, y_val = self._prepare_xy(df_val)

        # Get averaged predictions
        preds_list = []
        for model in models:
            preds_list.append(model.predict(X_val))

        avg_preds = np.mean(preds_list, axis=0)

        # Grid Search
        thresholds = np.arange(0.01, 1.00, 0.01)
        best_mcc = -1.0
        best_thresh = 0.5

        for thresh in thresholds:
            y_pred_bin = (avg_preds > thresh).astype(int)
            mcc = compute_mcc(y_val, y_pred_bin)
            if mcc > best_mcc:
                best_mcc = mcc
                best_thresh = thresh

        print(f"Best Threshold: {best_thresh:.2f}")
        print(f"Best Validation MCC: {best_mcc:.16f}")

        # Save threshold
        thresh_path = os.path.join(MODEL_OUTPUT_DIR, "best_threshold.npy")
        np.save(thresh_path, np.array([best_thresh]))

        return best_thresh

    def run(self):
        """
        Executes the full pipeline.
        """
        # 1. Load Data
        print("Loading processed datasets...")
        sample_size = 5000 if self.debug else None

        # Train set (Gated)
        df_train = self.data_factory.get_processed_dataset(
            mode="train", load_cached_data=True, sample_size=sample_size
        )

        # Cite debug_lesson_4: Ensure consistent index (RangeIndex) to match potential cache behavior.
        df_train = df_train.reset_index(drop=True)

        # Validation set
        df_val = self.data_factory.get_processed_dataset(
            mode="val", load_cached_data=True, sample_size=sample_size
        )

        # 2. Scout Training
        scout_model = self.run_scout_training(df_train, df_val)

        # 3. Hard Negative Mining
        df_hard_neg = self.mine_hard_negatives(df_train, scout_model)

        # 4. Expert Training
        lgbm_expert, xgb_expert = self.run_expert_training(
            df_train, df_hard_neg, df_val
        )

        # 5. Threshold Optimization
        self.optimize_threshold([lgbm_expert, xgb_expert], df_val)

        print("\nPipeline execution completed successfully.")
