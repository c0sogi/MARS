import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
import joblib

from library.config import Config
from library.utils import seed_everything, kendall_tau_metric
from library.feature_engineering import TextVectorizer
from library.data_loader import NotebookProcessor, load_metadata


class Stage1Ridge:
    """
    Implements Stage 1 of the pipeline: Sparse Lexical Regression (Ridge).
    Generates OOF predictions for training data and final predictions for val/test data.
    """

    def __init__(self, config=Config):
        self.config = config
        self.vectorizer = TextVectorizer(config)
        self.model = Ridge(
            alpha=config.RIDGE_ALPHA,
            solver=config.RIDGE_SOLVER,
            random_state=config.SEED,
        )

    def _get_sparse_features(self, texts, is_train=False):
        """
        Fits (if train) and transforms texts using the TF-IDF vectorizer.
        Returns sparse matrix.
        """
        # Ensure vectorizer is fitted
        if is_train:
            self.vectorizer.fit(texts, load_cached_models=True)

        # Access the underlying TF-IDF vectorizer directly for sparse features
        if self.vectorizer.tfidf is None:
            # Attempt to load if not in memory (e.g. if is_train=False and fit wasn't called on this instance)
            self.vectorizer.load()

        return self.vectorizer.tfidf.transform(texts)

    def _convert_preds_to_ordering(self, df, pred_col):
        """
        Converts regression scores to cell order strings for metric calculation.
        """
        # Create a copy to avoid modifying original
        temp = df[["id", "cell_id", pred_col]].copy()
        temp = temp.sort_values(["id", pred_col])

        # Group by id and join cell_ids
        pred_orders = (
            temp.groupby("id")["cell_id"].apply(lambda x: " ".join(x)).reset_index()
        )
        pred_orders.columns = ["id", "cell_order"]
        return pred_orders

    def run(self, load_cached_preds=True):
        """
        Executes the Stage 1 pipeline.

        Args:
            load_cached_preds (bool): If True, attempts to load predictions from disk.

        Returns:
            tuple: (df_train_with_oof, df_val_with_preds, df_test_with_preds)
        """
        seed_everything(self.config.SEED)

        # Paths for cached predictions
        train_pred_path = self.config.TRAIN_RIDGE_OOF_PATH
        val_pred_path = self.config.VAL_RIDGE_PREDS_PATH
        test_pred_path = self.config.TEST_RIDGE_PREDS_PATH

        # 1. Check Cache
        if (
            load_cached_preds
            and os.path.exists(train_pred_path)
            and os.path.exists(val_pred_path)
            and os.path.exists(test_pred_path)
        ):
            print("Loading cached Stage 1 predictions...")
            df_train = pd.read_parquet(train_pred_path)
            df_val = pd.read_parquet(val_pred_path)
            df_test = pd.read_parquet(test_pred_path)
            return df_train, df_val, df_test

        print("Running Stage 1: Sparse Ridge Regression...")

        # 2. Load Data
        processor = NotebookProcessor(self.config)
        df_train = processor.load_data("train")
        df_val = processor.load_data("val")
        df_test = processor.load_data("test")

        # Prepare text data (handle NaNs)
        train_texts = df_train["source"].fillna("").astype(str)
        val_texts = df_val["source"].fillna("").astype(str)
        test_texts = df_test["source"].fillna("").astype(str)

        # 3. Vectorization (Sparse TF-IDF)
        print("Generating sparse features...")
        X_train = self._get_sparse_features(train_texts, is_train=True)
        X_val = self._get_sparse_features(val_texts, is_train=False)
        X_test = self._get_sparse_features(test_texts, is_train=False)

        # Targets
        y_train = df_train["norm_rank"].values
        groups = df_train["ancestor_id"].fillna(df_train["id"]).values

        # 4. Cross-Validation for OOF Predictions
        print("Starting GroupKFold Cross-Validation...")
        oof_preds = np.zeros(len(df_train))
        gkf = GroupKFold(n_splits=5)

        for fold, (train_idx, valid_idx) in enumerate(
            gkf.split(X_train, y_train, groups)
        ):
            X_fold_train, X_fold_val = X_train[train_idx], X_train[valid_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[valid_idx]

            self.model.fit(X_fold_train, y_fold_train)
            preds = self.model.predict(X_fold_val)
            oof_preds[valid_idx] = preds

            fold_mae = mean_absolute_error(y_fold_val, preds)
            print(f"Fold {fold+1} MAE: {fold_mae}")

        df_train["ridge_pred"] = oof_preds

        # 5. Full Training for Inference
        print("Retraining on full dataset for inference...")
        self.model.fit(X_train, y_train)

        val_preds = self.model.predict(X_val)
        test_preds = self.model.predict(X_test)

        df_val["ridge_pred"] = val_preds
        df_test["ridge_pred"] = test_preds

        # 6. Evaluation on Validation Set
        print("Evaluating on Validation Set...")
        # Load ground truth for validation
        val_meta = load_metadata("val")
        val_pred_orders = self._convert_preds_to_ordering(df_val, "ridge_pred")

        score = kendall_tau_metric(val_pred_orders, val_meta)
        print(f"Stage 1 Validation Kendall Tau: {score}")

        # 7. Save Predictions
        print("Saving predictions to cache...")
        os.makedirs(self.config.WORKING_DIR, exist_ok=True)

        df_train[["id", "cell_id", "ridge_pred"]].to_parquet(
            train_pred_path, index=False
        )
        df_val[["id", "cell_id", "ridge_pred"]].to_parquet(val_pred_path, index=False)
        df_test[["id", "cell_id", "ridge_pred"]].to_parquet(test_pred_path, index=False)

        # Save model
        joblib.dump(self.model, self.config.RIDGE_PATH)
        print(f"Model saved to {self.config.RIDGE_PATH}")

        return df_train, df_val, df_test


def run_stage1(load_cached_preds=True):
    """
    Helper function to instantiate and run the stage.
    """
    stage = Stage1Ridge(Config)
    return stage.run(load_cached_preds=load_cached_preds)
