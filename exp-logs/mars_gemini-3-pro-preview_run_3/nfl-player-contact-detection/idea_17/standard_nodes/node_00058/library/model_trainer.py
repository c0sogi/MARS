import pandas as pd
import numpy as np
import xgboost as xgb
import os
import gc
from library.config import Config
from library.utils import calc_mcc, optimize_thresholds


class StreamTrainer:
    """
    Manages the training and inference lifecycle for a specific data stream (A or B).
    Wraps XGBoost with custom undersampling and threshold optimization logic.
    """

    def __init__(self, stream_name):
        """
        Args:
            stream_name (str): Identifier for the stream (e.g., 'streamA', 'streamB').
        """
        self.stream_name = stream_name
        self.model = None
        self.best_threshold = 0.5
        self.feature_cols = []

        # Columns to exclude from features (Metadata)
        self.exclude_cols = [
            "contact_id",
            "game_play",
            "nfl_player_id_1",
            "nfl_player_id_2",
            "step",
            "datetime",
            "contact",
            "video_path_sideline",
            "video_path_endzone",
            "video_path_all29",
            "pair_id",
            "frame",
            "nfl_player_id",
        ]

    def _get_feature_cols(self, df):
        """Identifies feature columns by excluding metadata."""
        return [c for c in df.columns if c not in self.exclude_cols]

    def _undersample(self, df):
        """
        Performs Random Undersampling on the majority class (0) based on Config.NEG_POS_RATIO.
        """
        pos_mask = df["contact"] == 1
        pos_df = df[pos_mask]
        neg_df = df[~pos_mask]

        n_pos = len(pos_df)

        # Handle case where there are no positive samples (e.g., in small debug splits).
        # Cite debug_lesson_5: Explicitly Handle Single-Class Data Slices
        if n_pos == 0:
            return df

        n_neg = int(n_pos * Config.NEG_POS_RATIO)

        # Sample negatives if we have enough, otherwise keep all
        if n_neg < len(neg_df):
            neg_df = neg_df.sample(n=n_neg, random_state=Config.SEED)

        # Concatenate and shuffle
        sampled_df = pd.concat([pos_df, neg_df]).sample(
            frac=1, random_state=Config.SEED
        )
        return sampled_df

    def train(self, df_train, df_val):
        """
        Trains the XGBoost model with undersampling and early stopping.
        Optimizes the decision threshold on the validation set.

        Args:
            df_train (pd.DataFrame): Training data.
            df_val (pd.DataFrame): Validation data.
        """
        print(f"[{self.stream_name}] Starting training process...")

        # 1. Feature Selection
        self.feature_cols = self._get_feature_cols(df_train)
        print(f"[{self.stream_name}] Selected {len(self.feature_cols)} features.")

        # 2. Undersampling
        train_sampled = self._undersample(df_train)
        print(f"[{self.stream_name}] Undersampled train shape: {train_sampled.shape}")

        X_train = train_sampled[self.feature_cols]
        y_train = train_sampled["contact"]

        X_val = df_val[self.feature_cols]
        y_val = df_val["contact"]

        # 3. Initialize Model
        # Note: early_stopping_rounds is passed to fit()
        # Cite debug_lesson_18: Pass early_stopping_rounds to constructor, not fit
        xgb_params = Config.XGB_PARAMS.copy()
        if "early_stopping_rounds" not in xgb_params:
            xgb_params["early_stopping_rounds"] = Config.EARLY_STOPPING_ROUNDS

        # Explicitly set base_score to prevent crash on single-class data (all zeros)
        # where auto-estimation would result in 0.0 (invalid for logistic loss).
        xgb_params["base_score"] = 0.5

        self.model = xgb.XGBClassifier(**xgb_params)

        # 4. Fit Model
        print(f"[{self.stream_name}] Fitting XGBoost model...")
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=Config.VERBOSE_EVAL,
        )

        # 5. Optimize Threshold
        print(f"[{self.stream_name}] Optimizing threshold on validation set...")
        y_pred_proba = self.model.predict_proba(X_val)[:, 1]
        self.best_threshold, best_mcc = optimize_thresholds(y_val.values, y_pred_proba)

        # 6. Print Metrics (Full Precision)
        print(f"[{self.stream_name}] Final Validation MCC: {best_mcc}")
        print(f"[{self.stream_name}] Optimal Threshold: {self.best_threshold}")

        # Cleanup
        del X_train, y_train, X_val, y_val, train_sampled
        gc.collect()

    def predict(self, df_test):
        """
        Generates predictions for the test set using the trained model and optimal threshold.

        Args:
            df_test (pd.DataFrame): Test data.

        Returns:
            pd.DataFrame: DataFrame containing ['contact_id', 'contact', 'prob']
        """
        if df_test.empty:
            return pd.DataFrame(columns=["contact_id", "contact", "prob"])

        X_test = df_test[self.feature_cols]

        # Predict Probabilities
        probas = self.model.predict_proba(X_test)[:, 1]

        # Apply Threshold
        preds = (probas >= self.best_threshold).astype(int)

        # Format Result
        result = df_test[["contact_id"]].copy()
        result["contact"] = preds
        result["prob"] = probas

        return result

    def save_model(self, filename):
        """Saves the XGBoost model to a file."""
        if self.model:
            path = os.path.join(Config.WORKING_DIR, filename)
            self.model.save_model(path)
            print(f"[{self.stream_name}] Model saved to {path}")

    def load_model(self, filename):
        """Loads the XGBoost model from a file."""
        path = os.path.join(Config.WORKING_DIR, filename)
        if os.path.exists(path):
            self.model = xgb.XGBClassifier(**Config.XGB_PARAMS)
            self.model.load_model(path)
            print(f"[{self.stream_name}] Model loaded from {path}")
        else:
            print(f"[{self.stream_name}] Model file not found at {path}")


def generate_submission(preds_a, preds_b, output_path=Config.SUBMISSION_PATH):
    """
    Combines predictions from Stream A and Stream B, handles duplicates,
    and saves the final submission CSV.

    Args:
        preds_a (pd.DataFrame): Predictions from Stream A.
        preds_b (pd.DataFrame): Predictions from Stream B.
        output_path (str): Path to save the submission file.
    """
    print("Generating submission file...")

    # Concatenate predictions
    full_preds = pd.concat([preds_a, preds_b], axis=0)

    # Ensure no duplicates (though streams should be disjoint)
    full_preds = full_preds.drop_duplicates(subset=["contact_id"])

    # Select required columns
    submission = full_preds[["contact_id", "contact"]]

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)

    print(f"Submission saved to {output_path}. Shape: {submission.shape}")
