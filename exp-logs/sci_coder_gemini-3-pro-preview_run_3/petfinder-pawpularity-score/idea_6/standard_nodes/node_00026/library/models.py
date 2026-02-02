import os
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.base import clone

from library.config import Config
from library.utils import set_seed


class Level1Estimators:
    """
    Manages the Level 1 base learners for the stacking ensemble.
    Includes SVR, Ridge Regression, and LightGBM.
    """

    def __init__(self):
        set_seed(Config.SEED)

        # Define base models with robust hyperparameters
        # SVR: RBF kernel is effective for high-dimensional dense features
        # Ridge: Handles multicollinearity well
        # LightGBM: Gradient boosting for non-linear interactions
        self.base_models_def = [
            ("svr", SVR(kernel="rbf", C=20.0, epsilon=0.1, gamma="scale")),
            (
                "et",
                ExtraTreesRegressor(
                    n_estimators=1000,
                    max_depth=32,
                    min_samples_leaf=10,
                    max_features="sqrt",
                    random_state=Config.SEED,
                    n_jobs=Config.NUM_WORKERS,
                ),
            ),
            (
                "lgbm",
                lgb.LGBMRegressor(
                    n_estimators=1000,
                    learning_rate=0.02,
                    num_leaves=31,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=Config.SEED,
                    n_jobs=Config.NUM_WORKERS,
                    verbose=-1,
                ),
            ),
        ]

        # Storage for models trained on the full dataset
        self.fitted_models_full = []

    def get_oof_predictions(self, X, y):
        """
        Performs K-Fold Cross-Validation to generate Out-of-Fold (OOF) predictions.

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training targets.

        Returns:
            np.ndarray: OOF predictions matrix of shape (n_samples, n_models).
        """
        n_samples = X.shape[0]
        n_models = len(self.base_models_def)
        oof_preds = np.zeros((n_samples, n_models))

        kf = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)

        print(f"Starting Level 1 Cross-Validation ({Config.N_FOLDS} folds)...")

        # Track metrics
        model_rmses = {name: [] for name, _ in self.base_models_def}

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            for model_idx, (name, model_template) in enumerate(self.base_models_def):
                # Clone model to ensure fresh training
                model = clone(model_template)

                # Fit model
                # Note: LightGBM verbose is handled in init
                model.fit(X_train, y_train)

                # Predict
                val_preds = model.predict(X_val)

                # Store OOF
                oof_preds[val_idx, model_idx] = val_preds

                # Calculate metric
                rmse = np.sqrt(mean_squared_error(y_val, val_preds))
                model_rmses[name].append(rmse)

        # Report average RMSE per model
        print("Level 1 CV Results (RMSE):")
        for name, scores in model_rmses.items():
            print(f"  {name}: {np.mean(scores)}")

        return oof_preds

    def fit_all(self, X, y):
        """
        Retrains all base models on the full training dataset.

        Args:
            X (np.ndarray): Full training features.
            y (np.ndarray): Full training targets.
        """
        print("Retraining Level 1 models on full dataset...")
        self.fitted_models_full = []

        for name, model_template in self.base_models_def:
            model = clone(model_template)
            model.fit(X, y)
            self.fitted_models_full.append((name, model))

        # Save models to disk for persistence
        save_path = os.path.join(Config.WORKING_DIR, "level1_models.joblib")
        joblib.dump(self.fitted_models_full, save_path)
        print(f"Level 1 models saved to {save_path}")

    def predict(self, X):
        """
        Generates predictions using the models trained on the full dataset.

        Args:
            X (np.ndarray): Test features.

        Returns:
            np.ndarray: Matrix of predictions (n_samples, n_models).
        """
        if not self.fitted_models_full:
            raise ValueError(
                "Models have not been fitted on full data yet. Call fit_all() first."
            )

        preds_list = []
        for name, model in self.fitted_models_full:
            preds = model.predict(X)
            preds_list.append(preds)

        return np.column_stack(preds_list)


class StackingMetaLearner:
    """
    Level 2 Meta-Learner (Linear Regression) that aggregates base model predictions.
    """

    def __init__(self):
        set_seed(Config.SEED)
        self.model = LinearRegression()

    def fit(self, X_oof, y):
        """
        Fits the meta-learner on OOF predictions.

        Args:
            X_oof (np.ndarray): OOF predictions from Level 1 (n_samples, n_models).
            y (np.ndarray): Target values.
        """
        print("Training Level 2 Meta-Learner...")
        self.model.fit(X_oof, y)

        # Evaluate on training data (OOF)
        preds = self.model.predict(X_oof)
        rmse = np.sqrt(mean_squared_error(y, preds))
        print(f"Level 2 (Stacking) OOF RMSE: {rmse}")
        print(f"Meta-Learner Coefficients: {self.model.coef_}")
        print(f"Meta-Learner Intercept: {self.model.intercept_}")

        # Save model
        save_path = os.path.join(Config.WORKING_DIR, "meta_learner.joblib")
        joblib.dump(self.model, save_path)

    def predict(self, X_base_preds):
        """
        Generates final predictions.

        Args:
            X_base_preds (np.ndarray): Predictions from Level 1 models on test set.

        Returns:
            np.ndarray: Final aggregated predictions.
        """
        return self.model.predict(X_base_preds)


def generate_submission(ids, predictions, output_path=None):
    """
    Formats predictions and saves the submission CSV.

    Args:
        ids (np.ndarray): Array of test IDs.
        predictions (np.ndarray): Array of predicted Pawpularity scores.
        output_path (str, optional): Path to save the CSV. Defaults to Config.SUBMISSION_PATH.
    """
    # Cite debug_lesson_8: Resolve default arguments at runtime to respect Config updates
    if output_path is None:
        output_path = Config.SUBMISSION_PATH

    # Ensure predictions are within valid range [1, 100]
    predictions = np.clip(predictions, 1.0, 100.0)

    df = pd.DataFrame({"Id": ids, "Pawpularity": predictions})

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print(df.head())
