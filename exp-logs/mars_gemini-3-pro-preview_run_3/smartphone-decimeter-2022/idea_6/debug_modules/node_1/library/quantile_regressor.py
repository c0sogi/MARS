import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from library.config import (
    LGBM_PARAMS,
    QUANTILES,
    EARLY_STOPPING_ROUNDS,
    VERBOSE_EVAL,
    WORKING_DIR,
)


class QuantileLGBMWrapper:
    """
    Wrapper for LightGBM Quantile Regression.
    Trains separate boosters for each specified quantile to estimate residuals and uncertainty.
    """

    def __init__(self, model_path_base, params=None, quantiles=None):
        """
        Initialize the wrapper.

        Args:
            model_path_base (str): Base path for saving/loading model files.
                                   Suffixes like '_q0.5.txt' will be appended.
            params (dict, optional): LightGBM parameters. Defaults to config.LGBM_PARAMS.
            quantiles (list, optional): List of quantiles to predict. Defaults to config.QUANTILES.
        """
        self.model_path_base = model_path_base
        self.params = params if params else LGBM_PARAMS.copy()
        self.quantiles = quantiles if quantiles else QUANTILES
        self.models = {}

    def _get_model_path(self, quantile):
        """Construct the file path for a specific quantile model."""
        return f"{self.model_path_base}_q{quantile}.txt"

    def fit(self, X_train, y_train, X_val, y_val, feature_names=None):
        """
        Train models for all configured quantiles using a fixed train/val split.
        Saves trained boosters to disk.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training targets.
            X_val (pd.DataFrame): Validation features.
            y_val (pd.Series): Validation targets.
            feature_names (list, optional): List of feature names.
        """
        print(f"\nTraining models for base path: {self.model_path_base}")

        for q in self.quantiles:
            print(f"  Training Quantile: {q}")

            # Update params for specific quantile objective
            current_params = self.params.copy()
            current_params["alpha"] = q

            # Create datasets
            dtrain = lgb.Dataset(
                X_train,
                label=y_train,
                feature_name=feature_names if feature_names else "auto",
            )
            dval = lgb.Dataset(
                X_val,
                label=y_val,
                reference=dtrain,
                feature_name=feature_names if feature_names else "auto",
            )

            # Callbacks for early stopping and logging
            callbacks = [
                lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS),
                lgb.log_evaluation(period=VERBOSE_EVAL),
            ]

            # Train
            model = lgb.train(
                current_params,
                dtrain,
                valid_sets=[dtrain, dval],
                valid_names=["train", "valid"],
                callbacks=callbacks,
            )

            # Save model to disk
            save_path = self._get_model_path(q)
            model.save_model(save_path)
            self.models[q] = model
            print(f"    Model saved to {save_path}")

    def train_group_kfold(self, X, y, groups, n_folds=5, feature_names=None):
        """
        Perform GroupKFold cross-validation to evaluate model performance.
        Prints metrics for each fold and the average.

        Args:
            X (pd.DataFrame): Features.
            y (pd.Series): Targets.
            groups (pd.Series): Group labels (e.g., drive_id) for splitting.
            n_folds (int): Number of folds.
            feature_names (list, optional): List of feature names.
        """
        gkf = GroupKFold(n_splits=n_folds)

        fold_metrics = {q: [] for q in self.quantiles}

        print(
            f"\nStarting GroupKFold CV ({n_folds} folds) for {self.model_path_base}..."
        )

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            print(f"\n--- Fold {fold + 1} ---")

            X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
            X_val_fold, y_val_fold = X.iloc[val_idx], y.iloc[val_idx]

            for q in self.quantiles:
                current_params = self.params.copy()
                current_params["alpha"] = q

                dtrain = lgb.Dataset(
                    X_train_fold,
                    label=y_train_fold,
                    feature_name=feature_names if feature_names else "auto",
                )
                dval = lgb.Dataset(
                    X_val_fold,
                    label=y_val_fold,
                    reference=dtrain,
                    feature_name=feature_names if feature_names else "auto",
                )

                # Train with reduced verbosity for CV
                model = lgb.train(
                    current_params,
                    dtrain,
                    valid_sets=[dval],
                    callbacks=[
                        lgb.early_stopping(
                            stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False
                        )
                    ],
                )

                # Evaluate
                preds = model.predict(X_val_fold)
                loss = self._quantile_loss(y_val_fold, preds, q)
                fold_metrics[q].append(loss)

                print(f"  Quantile {q} | Val Loss: {loss:.10f}")

        print("\n--- CV Summary ---")
        for q in self.quantiles:
            avg_loss = np.mean(fold_metrics[q])
            print(f"Quantile {q} Average Loss: {avg_loss:.10f}")

    def predict(self, X):
        """
        Generate predictions for all configured quantiles.
        Loads models from disk if they are not currently in memory.

        Args:
            X (pd.DataFrame): Input features.

        Returns:
            pd.DataFrame: DataFrame where columns are the quantiles (e.g., 0.1, 0.5, 0.9).
        """
        predictions = {}
        for q in self.quantiles:
            # Load model if not present
            if q not in self.models:
                path = self._get_model_path(q)
                if os.path.exists(path):
                    self.models[q] = lgb.Booster(model_file=path)
                else:
                    raise FileNotFoundError(
                        f"Model for quantile {q} not found at {path}. Call fit() first."
                    )

            predictions[q] = self.models[q].predict(X)

        return pd.DataFrame(predictions)

    def predict_with_uncertainty(self, X):
        """
        Predict the median residual and the uncertainty (Inter-Quantile Range).

        Args:
            X (pd.DataFrame): Input features.

        Returns:
            tuple: (median_predictions, uncertainty_values)
                   median_predictions: np.array of the 0.5 quantile predictions.
                   uncertainty_values: np.array of (q0.9 - q0.1) predictions.
        """
        preds_df = self.predict(X)

        # Ensure required quantiles exist
        if 0.5 not in preds_df.columns:
            raise ValueError("Median quantile (0.5) is missing from predictions.")
        if 0.1 not in preds_df.columns or 0.9 not in preds_df.columns:
            raise ValueError(
                "Uncertainty quantiles (0.1, 0.9) are missing from predictions."
            )

        median_pred = preds_df[0.5].values
        uncertainty = preds_df[0.9] - preds_df[0.1]

        # Ensure uncertainty is non-negative (can happen in rare crossing cases)
        uncertainty = np.maximum(uncertainty, 0.0)

        return median_pred, uncertainty.values

    @staticmethod
    def _quantile_loss(y_true, y_pred, quantile):
        """
        Calculate the Pinball (Quantile) Loss.
        """
        residual = y_true - y_pred
        return np.mean(np.maximum(quantile * residual, (quantile - 1) * residual))
