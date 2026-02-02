import numpy as np
import pandas as pd
import xgboost as xgb
import os
import gc
from library.config import Config
from library.utils import compute_mcc


class StreamModel:
    """
    Manages the training and inference of the dual-stream XGBoost models
    (Player-Player and Player-Ground) for a specific feature stream.
    """

    def __init__(self, stream_name: str):
        """
        Args:
            stream_name (str): Identifier for the stream (e.g., 'tracking', 'helmets').
        """
        self.stream_name = stream_name
        self.model_pp = None  # Player-Player Interaction Model
        self.model_pg = None  # Player-Ground Interaction Model
        self.config = Config

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        meta_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        meta_val: pd.DataFrame,
    ):
        """
        Trains the internal sub-models (PP and PG) with undersampling and early stopping.
        """
        print(f"[{self.stream_name.upper()}] Training Start...")

        # 1. Split Data into Player-Player (PP) and Player-Ground (PG)
        train_pp_idx, train_pg_idx = self._get_split_indices(meta_train)
        val_pp_idx, val_pg_idx = self._get_split_indices(meta_val)

        # 2. Train Player-Player Model
        print(f"[{self.stream_name.upper()}] Training Player-Player Model...")
        self.model_pp = self._train_sub_model(
            X_train, y_train, train_pp_idx, X_val, y_val, val_pp_idx, suffix="PP"
        )

        # 3. Train Player-Ground Model
        print(f"[{self.stream_name.upper()}] Training Player-Ground Model...")
        self.model_pg = self._train_sub_model(
            X_train, y_train, train_pg_idx, X_val, y_val, val_pg_idx, suffix="PG"
        )

        print(f"[{self.stream_name.upper()}] Training Complete.")

    def predict_proba(self, X: pd.DataFrame, meta: pd.DataFrame) -> np.ndarray:
        """
        Generates probability predictions for the input data.

        Args:
            X (pd.DataFrame): Feature matrix.
            meta (pd.DataFrame): Metadata containing player IDs to determine split.

        Returns:
            np.ndarray: Probability array of shape (n_samples,).
        """
        n_samples = len(X)
        preds = np.zeros(n_samples, dtype=np.float32)

        pp_idx, pg_idx = self._get_split_indices(meta)

        # Predict PP
        if len(pp_idx) > 0:
            if self.model_pp is not None:
                dtest_pp = xgb.DMatrix(X.iloc[pp_idx])
                preds[pp_idx] = self.model_pp.predict(dtest_pp)
            else:
                # Fallback if model wasn't trained (e.g. no data)
                preds[pp_idx] = 0.0

        # Predict PG
        if len(pg_idx) > 0:
            if self.model_pg is not None:
                dtest_pg = xgb.DMatrix(X.iloc[pg_idx])
                preds[pg_idx] = self.model_pg.predict(dtest_pg)
            else:
                preds[pg_idx] = 0.0

        return preds

    def _get_split_indices(self, meta: pd.DataFrame):
        """
        Returns indices for Player-Player and Player-Ground rows.
        Ground is identified by NaN in nfl_player_id_2 (result of numeric coercion).
        """
        # Note: DataManager loads parquet where nfl_player_id_2 is float with NaNs for 'G'
        is_ground = meta["nfl_player_id_2"].isna()

        idx_pg = np.where(is_ground)[0]
        idx_pp = np.where(~is_ground)[0]

        return idx_pp, idx_pg

    def _train_sub_model(self, X, y, train_idx, X_val, y_val, val_idx, suffix):
        """
        Helper to train a specific sub-model (PP or PG).
        """
        if len(train_idx) == 0:
            print(f"[{self.stream_name.upper()}-{suffix}] No training data available.")
            return None

        # Subset Data
        X_sub = X.iloc[train_idx]
        y_sub = y[train_idx]

        # Apply Undersampling
        X_res, y_res = self._apply_undersampling(X_sub, y_sub)

        # Create DMatrix
        dtrain = xgb.DMatrix(X_res, label=y_res)

        # Validation Data
        evals = [(dtrain, "train")]
        dval = None

        if len(val_idx) > 0:
            X_v_sub = X_val.iloc[val_idx]
            y_v_sub = y_val[val_idx]
            dval = xgb.DMatrix(X_v_sub, label=y_v_sub)
            evals.append((dval, "eval"))

        # Train
        params = self.config.XGB_PARAMS.copy()

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=params["n_estimators"],
            evals=evals,
            early_stopping_rounds=self.config.EARLY_STOPPING_ROUNDS,
            verbose_eval=self.config.VERBOSE_EVAL,
        )

        # Log Metrics
        if dval:
            val_preds = model.predict(dval)
            # Binary prediction for MCC (using 0.5 temporarily)
            val_preds_bin = (val_preds >= 0.5).astype(int)
            mcc = compute_mcc(y_v_sub, val_preds_bin)
            print(
                f"[{self.stream_name.upper()}-{suffix}] Best Iteration: {model.best_iteration}"
            )
            print(f"[{self.stream_name.upper()}-{suffix}] Validation MCC: {mcc}")

        return model

    def _apply_undersampling(self, X, y):
        """
        Undersamples the negative class based on Config.NEG_POS_RATIO.
        """
        # Identify classes
        pos_mask = y == 1
        neg_mask = y == 0

        pos_indices = np.where(pos_mask)[0]
        neg_indices = np.where(neg_mask)[0]

        n_pos = len(pos_indices)

        # Handle edge case with no positives
        if n_pos == 0:
            return X, y

        n_neg_keep = int(n_pos * self.config.NEG_POS_RATIO)

        if len(neg_indices) > n_neg_keep:
            # Randomly sample negatives
            np.random.seed(self.config.SEED)
            neg_indices_sampled = np.random.choice(
                neg_indices, n_neg_keep, replace=False
            )
        else:
            neg_indices_sampled = neg_indices

        # Combine
        final_indices = np.concatenate([pos_indices, neg_indices_sampled])
        np.random.shuffle(final_indices)

        return X.iloc[final_indices], y[final_indices]

    def save(self, directory):
        """Saves the models to the specified directory."""
        os.makedirs(directory, exist_ok=True)
        if self.model_pp:
            self.model_pp.save_model(
                os.path.join(directory, f"{self.stream_name}_pp.json")
            )
        if self.model_pg:
            self.model_pg.save_model(
                os.path.join(directory, f"{self.stream_name}_pg.json")
            )

    def load(self, directory):
        """Loads the models from the specified directory."""
        pp_path = os.path.join(directory, f"{self.stream_name}_pp.json")
        pg_path = os.path.join(directory, f"{self.stream_name}_pg.json")

        if os.path.exists(pp_path):
            self.model_pp = xgb.Booster()
            self.model_pp.load_model(pp_path)

        if os.path.exists(pg_path):
            self.model_pg = xgb.Booster()
            self.model_pg.load_model(pg_path)
