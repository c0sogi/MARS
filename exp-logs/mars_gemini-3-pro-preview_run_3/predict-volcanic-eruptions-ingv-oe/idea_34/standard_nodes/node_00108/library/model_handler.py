import os
import lightgbm as lgb
import pandas as pd
import numpy as np
from library.config import Config


class LGBMRegressorWrapper:
    """
    Wrapper for LightGBM Regressor to handle training, prediction, and submission generation
    specifically tailored for the seismic eruption prediction task.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = None
        self.params = {
            "num_leaves": cfg.NUM_LEAVES,
            "learning_rate": cfg.LEARNING_RATE,
            "objective": cfg.OBJECTIVE,
            "metric": cfg.METRIC,
            "feature_fraction": cfg.FEATURE_FRACTION,
            "bagging_fraction": cfg.BAGGING_FRACTION,
            "bagging_freq": cfg.BAGGING_FREQ,
            "verbosity": cfg.VERBOSITY,
            "seed": cfg.SEED,
            "n_jobs": -1,
        }

    def fit(self, X_train, y_train, X_val, y_val):
        """
        Trains the LightGBM model with early stopping.

        Args:
            X_train: Training features.
            y_train: Training target.
            X_val: Validation features.
            y_val: Validation target.
        """
        print(
            f"Initializing LightGBM training with params: leaves={self.cfg.NUM_LEAVES}, lr={self.cfg.LEARNING_RATE}"
        )

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        callbacks = [
            lgb.early_stopping(
                stopping_rounds=self.cfg.EARLY_STOPPING_ROUNDS, verbose=True
            ),
            lgb.log_evaluation(period=100),
        ]

        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.cfg.N_ESTIMATORS,
            valid_sets=[train_data, val_data],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

        # Print full precision metric
        if self.model.best_score:
            val_score = self.model.best_score["valid"][self.cfg.METRIC]
            print(f"Final Best Validation {self.cfg.METRIC}: {val_score}")

    def predict(self, X):
        """
        Generates predictions using the trained model.

        Args:
            X: Features to predict on.

        Returns:
            np.ndarray: Predicted values.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        return self.model.predict(X, num_iteration=self.model.best_iteration)

    def save_model(self, path):
        """Saves the trained booster to a file."""
        if self.model:
            self.model.save_model(path)
        else:
            print("No model to save.")

    def load_model(self, path):
        """Loads a booster from a file."""
        if os.path.exists(path):
            self.model = lgb.Booster(model_file=path)
        else:
            raise FileNotFoundError(f"Model file not found at {path}")

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
