import os
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from library.config import Config


class LGBMRegressorWrapper:
    """
    Wrapper for LightGBM Regressor to handle training, prediction, and submission generation
    specifically tailored for the seismic eruption prediction task.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.models = []
        self.params = {
            "num_leaves": cfg.NUM_LEAVES,
            "learning_rate": cfg.LEARNING_RATE,
            "objective": cfg.OBJECTIVE,
            "metric": cfg.METRIC,
            "lambda_l1": cfg.LAMBDA_L1,
            "lambda_l2": cfg.LAMBDA_L2,
            "feature_fraction": cfg.FEATURE_FRACTION,
            "bagging_fraction": cfg.BAGGING_FRACTION,
            "bagging_freq": cfg.BAGGING_FREQ,
            "verbosity": cfg.VERBOSITY,
            "seed": cfg.SEED,
            "n_jobs": -1,
        }

    def fit(self, X_train, y_train, X_val, y_val):
        """
        Trains a LightGBM ensemble using K-Fold Cross-Validation on X_train.
        X_val is used purely for external monitoring, not for training/early-stopping of folds.

        Args:
            X_train: Training features.
            y_train: Training target.
            X_val: Validation features (hold-out).
            y_val: Validation target (hold-out).
        """
        print(
            f"Initializing LightGBM Ensemble ({self.cfg.N_FOLDS} folds) with params: leaves={self.cfg.NUM_LEAVES}, lr={self.cfg.LEARNING_RATE}"
        )

        kf = KFold(n_splits=self.cfg.N_FOLDS, shuffle=True, random_state=self.cfg.SEED)

        # Reset models
        self.models = []

        # Ensure indices are reset for correct splitting
        X_t = X_train.reset_index(drop=True)
        y_t = y_train.reset_index(drop=True)

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_t, y_t)):
            print(f"--- Training Fold {fold + 1}/{self.cfg.N_FOLDS} ---")

            X_fold_train, y_fold_train = X_t.iloc[train_idx], y_t.iloc[train_idx]
            X_fold_val, y_fold_val = X_t.iloc[val_idx], y_t.iloc[val_idx]

            train_data = lgb.Dataset(X_fold_train, label=y_fold_train)
            val_data = lgb.Dataset(X_fold_val, label=y_fold_val, reference=train_data)

            callbacks = [
                lgb.early_stopping(
                    stopping_rounds=self.cfg.EARLY_STOPPING_ROUNDS, verbose=False
                ),
                lgb.log_evaluation(period=0),  # Silence per-fold logs to avoid clutter
            ]

            model = lgb.train(
                self.params,
                train_data,
                num_boost_round=self.cfg.N_ESTIMATORS,
                valid_sets=[train_data, val_data],
                valid_names=["train", "valid"],
                callbacks=callbacks,
            )

            best_score = model.best_score["valid"][self.cfg.METRIC]
            print(f"Fold {fold + 1} Best {self.cfg.METRIC}: {best_score}")
            self.models.append(model)

    def predict(self, X):
        """
        Generates predictions using the trained ensemble (averaging).

        Args:
            X: Features to predict on.

        Returns:
            np.ndarray: Predicted values.
        """
        if not self.models:
            raise ValueError("Model has not been trained yet.")

        preds = np.zeros(len(X))
        for model in self.models:
            preds += model.predict(X, num_iteration=model.best_iteration)

        return preds / len(self.models)

    def save_model(self, path):
        """Saves the trained boosters (not implemented for ensemble in this snippet)."""
        pass

    def load_model(self, path):
        """Loads a booster (not implemented for ensemble in this snippet)."""
        pass

    def generate_submission(self, X_test, test_ids):
        """
        Generates predictions for the test set and saves the submission file.

        Args:
            X_test: Test features.
            test_ids: Series or list of segment IDs corresponding to X_test.
        """
        print("Generating predictions for test set...")
        preds = self.predict(X_test)

        sub_df = pd.DataFrame({"segment_id": test_ids, "time_to_eruption": preds})

        os.makedirs(self.cfg.SUBMISSION_DIR, exist_ok=True)
        sub_df.to_csv(self.cfg.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {self.cfg.SUBMISSION_PATH}")
