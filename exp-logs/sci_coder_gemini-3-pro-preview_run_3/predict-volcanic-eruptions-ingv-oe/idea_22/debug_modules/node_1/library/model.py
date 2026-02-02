import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from library.config import Config
from library.utils import setup_logger
from library.dataset import VolcanoDataset


class ModelTrainer:
    """
    Encapsulates the training and inference logic for the Volcano Eruption Prediction.
    Implements a single LightGBM Regressor with L2/Huber loss and strict regularization
    as per the 'Hybrid-Transform Orthogonal Decomposition' strategy.
    """

    def __init__(self):
        self.logger = setup_logger("ModelTrainer")
        self.dataset = VolcanoDataset()
        self.model = None

        # Ensure model artifact directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        self.model_path = os.path.join(Config.WORKING_DIR, "lgbm_model.txt")

    def train(self, load_cached_data: bool = True, limit: int = None):
        """
        Trains the LightGBM model using the configured parameters and dataset.

        Args:
            load_cached_data (bool): Whether to use cached feature data.
            limit (int): Limit the number of samples for debugging.
        """
        self.logger.info("Starting training process...")

        # 1. Load Data
        X_train, y_train = self.dataset.get_train_data(
            load_cached_data=load_cached_data, limit=limit
        )
        X_val, y_val = self.dataset.get_val_data(
            load_cached_data=load_cached_data, limit=limit
        )

        self.logger.info(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")

        # 2. Prepare LightGBM Datasets
        train_set = lgb.Dataset(X_train, label=y_train)
        val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

        # 3. Configure Callbacks
        callbacks = [
            lgb.early_stopping(stopping_rounds=Config.EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=Config.VERBOSE_EVAL),
        ]

        # 4. Train Model
        # Using L2 Loss (regression) as defined in Config to ensure informative gradients
        self.logger.info(f"Training with params: {Config.LGBM_PARAMS}")

        self.model = lgb.train(
            params=Config.LGBM_PARAMS,
            train_set=train_set,
            valid_sets=[train_set, val_set],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

        # 5. Log Results
        best_score = self.model.best_score["valid"]["mae"]
        self.logger.info(
            f"Training finished. Best Iteration: {self.model.best_iteration}"
        )
        self.logger.info(f"Best Validation MAE (Full Precision): {best_score}")

        # 6. Save Model
        self.model.save_model(self.model_path)
        self.logger.info(f"Model saved to {self.model_path}")

        return self.model

    def generate_submission(self, load_cached_data: bool = True, limit: int = None):
        """
        Generates predictions for the test set and saves the submission file.

        Args:
            load_cached_data (bool): Whether to use cached feature data.
            limit (int): Limit the number of samples for debugging.
        """
        if self.model is None:
            # Try to load from disk if not in memory
            if os.path.exists(self.model_path):
                self.logger.info(f"Loading model from {self.model_path}")
                self.model = lgb.Booster(model_file=self.model_path)
            else:
                raise RuntimeError("Model not trained and no saved model found.")

        self.logger.info("Generating submission...")

        # 1. Load Test Data
        X_test, segment_ids = self.dataset.get_test_data(
            load_cached_data=load_cached_data, limit=limit
        )

        # 2. Predict
        # Predict using the best iteration found during training
        predictions = self.model.predict(
            X_test, num_iteration=self.model.best_iteration
        )

        # 3. Create Submission DataFrame
        submission_df = pd.DataFrame(
            {"segment_id": segment_ids, "time_to_eruption": predictions}
        )

        # Ensure segment_id is int64 as per sample submission
        submission_df["segment_id"] = submission_df["segment_id"].astype(np.int64)

        # 4. Save
        # Ensure submission directory exists
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        self.logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
        self.logger.info(f"Submission shape: {submission_df.shape}")

        # Print head for verification
        print(submission_df.head())
