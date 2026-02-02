import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
import lightgbm as lgb

from library.config import (
    RIDGE_ALPHA,
    LGBM_PARAMS,
    WORKING_DIR,
    RANDOM_STATE,
    EARLY_STOPPING_ROUNDS,
    VERBOSE_EVAL,
)
from library.utils import seed_everything


class Stage1Ridge:
    """
    Stage 1: Sparse Lexical Regressor (The "Signpost" Model).
    Uses Ridge Regression on high-dimensional TF-IDF vectors.
    """

    def __init__(self, alpha=RIDGE_ALPHA):
        seed_everything(RANDOM_STATE)
        self.alpha = alpha
        self.model = Ridge(alpha=self.alpha, random_state=RANDOM_STATE)
        self.model_path = os.path.join(WORKING_DIR, "stage1_ridge_model.joblib")

    def fit(self, X, y):
        """
        Fits the Ridge model on the provided data.
        """
        print(f"Fitting Stage 1 Ridge Model (alpha={self.alpha})...")
        self.model.fit(X, y)

        # Save the fitted model
        os.makedirs(WORKING_DIR, exist_ok=True)
        joblib.dump(self.model, self.model_path)
        return self

    def predict(self, X):
        """
        Predicts using the fitted Ridge model.
        """
        return self.model.predict(X)

    def load_model(self):
        """
        Loads the model from disk if available.
        """
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            return True
        return False

    def get_oof_predictions(self, X, y, cell_ids, n_splits=5, load_cached_data=True):
        """
        Generates Out-Of-Fold (OOF) predictions using K-Fold Cross-Validation.
        Implements caching to store/retrieve OOF predictions.

        Args:
            X (scipy.sparse.csr_matrix): Feature matrix.
            y (array-like): Target values.
            cell_ids (array-like): Array of cell IDs corresponding to X rows.
            n_splits (int): Number of CV folds.
            load_cached_data (bool): Whether to load from disk if available.

        Returns:
            pd.DataFrame: DataFrame containing ['cell_id', 'ridge_rank'].
        """
        os.makedirs(WORKING_DIR, exist_ok=True)
        cache_path = os.path.join(WORKING_DIR, "stage1_oof_preds.parquet")

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                print(f"Loading cached Stage 1 OOF predictions from {cache_path}...")
                return pd.read_parquet(cache_path)
            except Exception as e:
                print(f"Failed to load OOF cache: {e}. Recomputing...")

        # 2. Compute from scratch
        print(f"Generating Stage 1 OOF predictions ({n_splits}-Fold CV)...")

        # Initialize storage
        oof_preds = np.zeros(len(y))

        # KFold
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val = X[val_idx]

            # Fit temporary model for this fold
            model = Ridge(alpha=self.alpha, random_state=RANDOM_STATE)
            model.fit(X_train, y_train)

            # Predict
            oof_preds[val_idx] = model.predict(X_val)

        # Create DataFrame
        df_oof = pd.DataFrame({"cell_id": cell_ids, "ridge_rank": oof_preds})

        # 3. Save to cache
        print(f"Saving Stage 1 OOF predictions to {cache_path}...")
        df_oof.to_parquet(cache_path, index=False)

        return df_oof


class Stage2LGBM:
    """
    Stage 2: Decoupled Multi-Resolution Gradient Booster (The "Refinement" Model).
    Uses LightGBM to refine predictions based on neighborhood features and interactions.
    """

    def __init__(self):
        seed_everything(RANDOM_STATE)
        self.params = LGBM_PARAMS.copy()
        self.model = lgb.LGBMRegressor(**self.params)
        self.model_path = os.path.join(WORKING_DIR, "stage2_lgbm_model.joblib")

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Fits the LightGBM model with Early Stopping.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training targets.
            X_val (pd.DataFrame, optional): Validation features.
            y_val (pd.Series, optional): Validation targets.
        """
        print("Fitting Stage 2 LightGBM Model...")

        eval_set = []
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]

        callbacks = [lgb.log_evaluation(period=VERBOSE_EVAL)]
        if eval_set:
            callbacks.append(
                lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=True)
            )

        self.model.fit(
            X_train, y_train, eval_set=eval_set, eval_metric="mae", callbacks=callbacks
        )

        # Print final metric if validation set was provided
        if self.model.best_score_:
            # best_score_ structure: {'valid_0': {'l1': 0.1234}}
            for valid_key, metrics in self.model.best_score_.items():
                for metric_name, score in metrics.items():
                    print(f"Best {valid_key} {metric_name}: {score}")

        # Save model
        os.makedirs(WORKING_DIR, exist_ok=True)
        joblib.dump(self.model, self.model_path)
        return self

    def predict(self, X):
        """
        Predicts using the fitted LightGBM model.
        """
        return self.model.predict(X)

    def load_model(self):
        """
        Loads the model from disk if available.
        """
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            return True
        return False
