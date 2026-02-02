import os
import numpy as np
import joblib
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from typing import Optional, Union

from library.config import Config


class RidgeStacker:
    """
    Stage 1: Sparse Lexical Regressor.
    Implements a Ridge Regression model on high-dimensional TF-IDF features.
    Handles Cross-Validation (OOF generation) and final model fitting.
    """

    def __init__(self):
        self.working_dir = Config.WORKING_DIR
        self.model_path = os.path.join(self.working_dir, "ridge_model.joblib")
        self.oof_path = os.path.join(self.working_dir, "stage1_oof_preds.npy")

        # Initialize Ridge Model with Config parameters
        self.model = Ridge(
            alpha=Config.RIDGE_ALPHA,
            solver=Config.RIDGE_SOLVER,
            random_state=Config.RANDOM_STATE,
        )

    def train_oof(
        self,
        X_sparse,
        y: np.ndarray,
        groups: np.ndarray,
        n_splits: int = 5,
        load_cached_data: bool = True,
    ) -> np.ndarray:
        """
        Performs Group K-Fold Cross Validation to generate Out-Of-Fold predictions.
        These predictions serve as unbiased features for the Stage 2 model.

        Args:
            X_sparse: Scipy sparse matrix of TF-IDF features.
            y: Target array (Normalized Ranks).
            groups: Array of ancestor_ids to strictly separate groups during CV.
            n_splits: Number of CV folds.
            load_cached_data: Whether to load OOF preds from disk if available.

        Returns:
            Numpy array of OOF predictions aligned with the input data.
        """
        os.makedirs(self.working_dir, exist_ok=True)

        # 1. Check Cache
        if load_cached_data and os.path.exists(self.oof_path):
            # print(f"Loading cached OOF predictions from {self.oof_path}")
            return np.load(self.oof_path)

        print(f"Starting Stage 1 OOF Training with {n_splits} folds...")

        # 2. Setup CV
        gkf = GroupKFold(n_splits=n_splits)
        oof_preds = np.zeros(len(y))

        # 3. Training Loop
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X_sparse, y, groups)):
            # Slice data
            X_train, y_train = X_sparse[train_idx], y[train_idx]
            X_val, y_val = X_sparse[val_idx], y[val_idx]

            # Fit
            self.model.fit(X_train, y_train)

            # Predict
            val_preds = self.model.predict(X_val)

            # Store
            oof_preds[val_idx] = val_preds

            # Metric
            mae = mean_absolute_error(y_val, val_preds)
            print(f"Fold {fold + 1} MAE: {mae}")

        # 4. Save Cache
        try:
            np.save(self.oof_path, oof_preds)
            # print(f"Saved OOF predictions to {self.oof_path}")
        except Exception as e:
            print(f"Warning: Failed to save OOF cache. Error: {e}")

        overall_mae = mean_absolute_error(y, oof_preds)
        print(f"Stage 1 Overall OOF MAE: {overall_mae}")

        return oof_preds

    def fit_final(self, X_sparse, y: np.ndarray):
        """
        Fits the Ridge model on the full dataset and saves it for inference.
        """
        print("Fitting Final Stage 1 Ridge Model on full dataset...")
        self.model.fit(X_sparse, y)

        os.makedirs(self.working_dir, exist_ok=True)
        joblib.dump(self.model, self.model_path)
        print(f"Stage 1 Model saved to {self.model_path}")

    def predict(self, X_sparse) -> np.ndarray:
        """
        Generates predictions using the trained model.
        Loads the model from disk if not currently fitted in memory.
        """
        # Check if model is fitted (Ridge stores coef_ after fitting)
        if not hasattr(self.model, "coef_"):
            if os.path.exists(self.model_path):
                # print(f"Loading Stage 1 model from {self.model_path}")
                self.model = joblib.load(self.model_path)
            else:
                raise FileNotFoundError(
                    "Ridge model not found. Call fit_final() first."
                )

        return self.model.predict(X_sparse)
