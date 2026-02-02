import os
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from library.config import Config


class Stage1Ridge:
    """
    Stage 1: Sparse Lexical Regressor (The "Signpost" Model).
    Wraps Ridge Regression to handle high-dimensional TF-IDF vectors.
    """

    def __init__(self):
        self.model = None
        self.config = Config.RIDGE_PARAMS

    def fit(self, X, y):
        """
        Fits the Ridge model on the provided data.
        Used for training on the full dataset before inference.
        """
        print("Training Stage 1 Ridge model on full dataset...")
        self.model = Ridge(**self.config)
        self.model.fit(X, y)

        # Calculate and print training metric
        preds = self.model.predict(X)
        mae = mean_absolute_error(y, preds)
        print(f"Stage 1 Training MAE: {mae}")

    def predict(self, X):
        """
        Predicts using the fitted Ridge model.
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        return self.model.predict(X)

    def get_oof_predictions(self, X, y, groups):
        """
        Generates Out-Of-Fold (OOF) predictions using GroupKFold.
        These predictions are used as features for Stage 2.
        """
        print(f"Generating Stage 1 OOF predictions with {Config.N_FOLDS} folds...")

        kf = GroupKFold(n_splits=Config.N_FOLDS)
        oof_preds = np.zeros(len(y))

        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y, groups)):
            # Split data
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            # Initialize and train fold model
            fold_model = Ridge(**self.config)
            fold_model.fit(X_train, y_train)

            # Predict
            val_preds = fold_model.predict(X_val)
            oof_preds[val_idx] = val_preds

            # Evaluate
            fold_mae = mean_absolute_error(y_val, val_preds)
            fold_scores.append(fold_mae)
            print(f"Fold {fold + 1} MAE: {fold_mae}")

        overall_mae = mean_absolute_error(y, oof_preds)
        print(f"Overall Stage 1 OOF MAE: {overall_mae}")

        return oof_preds

    def save(self, filepath):
        """Saves the model to disk."""
        joblib.dump(self.model, filepath)
        print(f"Stage 1 model saved to {filepath}")

    def load(self, filepath):
        """Loads the model from disk."""
        self.model = joblib.load(filepath)
        print(f"Stage 1 model loaded from {filepath}")


class Stage2LGBM:
    """
    Stage 2: Multi-View Anchor Gradient Booster (The "Refinement" Model).
    Wraps LightGBM Regressor to refine predictions using stacked features.
    """

    def __init__(self):
        self.model = None
        self.params = Config.LGBM_PARAMS.copy()

    def fit(self, X_train, y_train, X_val, y_val, feature_names=None):
        """
        Fits the LightGBM model with early stopping.
        """
        print("Training Stage 2 LightGBM model...")

        self.model = lgb.LGBMRegressor(**self.params)

        callbacks = [
            lgb.early_stopping(stopping_rounds=Config.LGBM_EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=Config.LGBM_VERBOSE_EVAL),
        ]

        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_names=["train", "valid"],
            eval_metric="mae",
            feature_name=feature_names if feature_names is not None else "auto",
            callbacks=callbacks,
        )

        # Print final validation score
        if self.model.best_score_ is not None:
            val_score = self.model.best_score_["valid"]["l1"]
            print(f"Best Validation MAE: {val_score}")

    def predict(self, X):
        """
        Predicts using the fitted LightGBM model.
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        return self.model.predict(X)

    def save(self, filepath):
        """Saves the model to disk."""
        joblib.dump(self.model, filepath)
        print(f"Stage 2 model saved to {filepath}")

    def load(self, filepath):
        """Loads the model from disk."""
        self.model = joblib.load(filepath)
        print(f"Stage 2 model loaded from {filepath}")
