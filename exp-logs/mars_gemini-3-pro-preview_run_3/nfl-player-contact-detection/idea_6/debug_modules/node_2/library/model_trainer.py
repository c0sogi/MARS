import pandas as pd
import numpy as np
import xgboost as xgb
import gc
from sklearn.metrics import matthews_corrcoef
from library.config import Config
from library.utils import setup_logger


class StreamModel:
    """
    Manages a dual-stream XGBoost model for Contact Detection.
    Internally separates Player-Player (PP) and Player-Ground (PG) interactions
    to train specialized classifiers for each type.
    """

    def __init__(self, name="StreamModel"):
        self.name = name
        self.config = Config
        self.logger = setup_logger(self.name)
        self.model_pp = None
        self.model_pg = None

        # Prepare parameters
        self.xgb_params = self.config.XGB_PARAMS.copy()
        # Extract fit-specific parameters that shouldn't be in __init__
        self.early_stopping_rounds = self.xgb_params.pop("early_stopping_rounds", 50)

    def _is_ground(self, df):
        """
        Robustly identifies Player-Ground rows.
        Checks for 'is_ground' column (Tracking FE) or 'nfl_player_id_2' == 'G' (Helmet FE).
        """
        if "is_ground" in df.columns:
            return df["is_ground"] == 1
        else:
            # Ensure string comparison
            return df["nfl_player_id_2"].astype(str) == "G"

    def _undersample(self, df, target_col):
        """
        Performs random undersampling on the majority class (0).
        Keeps all positives (1).
        """
        pos = df[df[target_col] == 1]
        neg = df[df[target_col] == 0]

        n_pos = len(pos)
        n_neg_keep = int(n_pos * self.config.UNDERSAMPLE_RATIO)

        # If we have fewer negatives than the ratio implies, keep all negatives
        if len(neg) > n_neg_keep:
            neg = neg.sample(n=n_neg_keep, random_state=self.config.SEED)

        sampled = pd.concat([pos, neg], axis=0)
        # Shuffle
        sampled = sampled.sample(frac=1, random_state=self.config.SEED).reset_index(
            drop=True
        )
        return sampled

    def train(self, train_df, val_df, feature_cols, target_col="contact"):
        """
        Trains the dual-stream model.

        Args:
            train_df (pd.DataFrame): Training data containing features and metadata.
            val_df (pd.DataFrame): Validation data.
            feature_cols (list): List of column names to use as features.
            target_col (str): Name of the target column.
        """
        self.logger.info(f"Starting training for {self.name}...")

        # 1. Undersample Training Data
        self.logger.info(
            f"Undersampling training data (Ratio {self.config.UNDERSAMPLE_RATIO}:1)..."
        )
        train_sampled = self._undersample(train_df, target_col)

        # 2. Split into PP and PG
        # Masks
        train_g_mask = self._is_ground(train_sampled)
        val_g_mask = self._is_ground(val_df)

        # Subsets
        X_train_pp = train_sampled[~train_g_mask][feature_cols]
        y_train_pp = train_sampled[~train_g_mask][target_col]

        X_train_pg = train_sampled[train_g_mask][feature_cols]
        y_train_pg = train_sampled[train_g_mask][target_col]

        X_val_pp = val_df[~val_g_mask][feature_cols]
        y_val_pp = val_df[~val_g_mask][target_col]

        X_val_pg = val_df[val_g_mask][feature_cols]
        y_val_pg = val_df[val_g_mask][target_col]

        self.logger.info(
            f"Train PP shape: {X_train_pp.shape}, PG shape: {X_train_pg.shape}"
        )
        self.logger.info(f"Val PP shape: {X_val_pp.shape}, PG shape: {X_val_pg.shape}")

        # 3. Train Player-Player Model
        self.logger.info("Training Player-Player (PP) Model...")
        self.model_pp = xgb.XGBClassifier(**self.xgb_params)
        self.model_pp.fit(
            X_train_pp,
            y_train_pp,
            eval_set=[(X_val_pp, y_val_pp)],
            early_stopping_rounds=self.early_stopping_rounds,
            verbose=False,
        )

        # Evaluate PP
        val_preds_pp = self.model_pp.predict(X_val_pp)
        mcc_pp = matthews_corrcoef(y_val_pp, val_preds_pp)
        print(f"[{self.name}] Validation MCC (PP): {mcc_pp}")

        # 4. Train Player-Ground Model
        self.logger.info("Training Player-Ground (PG) Model...")
        self.model_pg = xgb.XGBClassifier(**self.xgb_params)
        self.model_pg.fit(
            X_train_pg,
            y_train_pg,
            eval_set=[(X_val_pg, y_val_pg)],
            early_stopping_rounds=self.early_stopping_rounds,
            verbose=False,
        )

        # Evaluate PG
        val_preds_pg = self.model_pg.predict(X_val_pg)
        mcc_pg = matthews_corrcoef(y_val_pg, val_preds_pg)
        print(f"[{self.name}] Validation MCC (PG): {mcc_pg}")

        # 5. Overall Validation Score
        # We need to recombine predictions to score the full validation set
        all_preds = self.predict(val_df, feature_cols)
        # Convert probabilities to binary for MCC calculation (default threshold 0.5)
        # Note: Optimization of threshold happens later in the pipeline, but we print 0.5 metric here.
        binary_preds = (all_preds > 0.5).astype(int)
        total_mcc = matthews_corrcoef(val_df[target_col], binary_preds)

        print(f"[{self.name}] Total Validation MCC (Threshold 0.5): {total_mcc}")

        # Cleanup
        del train_sampled, X_train_pp, X_train_pg, X_val_pp, X_val_pg
        gc.collect()

    def predict(self, df, feature_cols):
        """
        Generates probabilities for the input DataFrame.

        Args:
            df (pd.DataFrame): Input data.
            feature_cols (list): Feature columns.

        Returns:
            np.ndarray: Array of probabilities (same length as df).
        """
        if self.model_pp is None or self.model_pg is None:
            raise RuntimeError("Models not trained yet.")

        # Initialize result array
        probs = np.zeros(len(df), dtype=np.float32)

        # Identify masks
        g_mask = self._is_ground(df)
        pp_mask = ~g_mask

        # Predict PP
        if pp_mask.any():
            X_pp = df.loc[pp_mask, feature_cols]
            # predict_proba returns [prob_0, prob_1]
            probs[pp_mask] = self.model_pp.predict_proba(X_pp)[:, 1]

        # Predict PG
        if g_mask.any():
            X_pg = df.loc[g_mask, feature_cols]
            probs[g_mask] = self.model_pg.predict_proba(X_pg)[:, 1]

        return probs

    @staticmethod
    def optimize_blending(y_true, pred_a, pred_b, n_trials=50):
        """
        Finds the optimal weight 'w' such that w*pred_a + (1-w)*pred_b maximizes MCC.
        Also finds the optimal decision threshold.

        Args:
            y_true (np.array): Ground truth labels.
            pred_a (np.array): Probabilities from Model A.
            pred_b (np.array): Probabilities from Model B.
            n_trials (int): Number of steps for weight search.

        Returns:
            tuple: (best_weight, best_threshold, best_mcc)
        """
        best_mcc = -1
        best_w = 0.5
        best_thresh = 0.5

        # Grid search for weight
        weights = np.linspace(0, 1, n_trials)

        # For threshold optimization, we check percentiles or fixed range
        thresholds = np.linspace(0.2, 0.8, 60)

        for w in weights:
            blended = w * pred_a + (1 - w) * pred_b

            # Vectorized threshold search for this weight
            for t in thresholds:
                preds = (blended > t).astype(int)
                score = matthews_corrcoef(y_true, preds)

                if score > best_mcc:
                    best_mcc = score
                    best_w = w
                    best_thresh = t

        return best_w, best_thresh, best_mcc
