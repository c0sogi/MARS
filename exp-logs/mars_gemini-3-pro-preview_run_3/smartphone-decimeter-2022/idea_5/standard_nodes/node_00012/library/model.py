import lightgbm as lgb
import pandas as pd
import numpy as np
import os
from library.config import Config
from library.utils import get_logger

logger = get_logger(__name__)


class LGBMRegressorWrapper:
    """
    Wrapper for LightGBM models to predict ENU residuals.
    Trains two separate models: one for East error, one for North error.
    """

    def __init__(self):
        # Make a copy to avoid modifying the global config dict
        self.params = Config.LGBM_PARAMS.copy()
        self.features = Config.FEATURES
        self.model_east = None
        self.model_north = None
        # Define specific paths for the two models
        self.model_path_east = os.path.join(Config.WORKING_DIR, "lgbm_east.txt")
        self.model_path_north = os.path.join(Config.WORKING_DIR, "lgbm_north.txt")

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        """
        Trains the East and North residual models using LightGBM.

        Args:
            train_df (pd.DataFrame): Training data containing features and targets.
            val_df (pd.DataFrame): Validation data containing features and targets.
        """
        logger.info("Preparing datasets for LightGBM training...")

        # Extract number of boosting rounds from params
        num_boost_round = self.params.pop("n_estimators", 2000)

        # Prepare Feature Matrices
        X_train = train_df[self.features]
        X_val = val_df[self.features]

        # ---------------------------------------------------------------------
        # Train East Model
        # ---------------------------------------------------------------------
        logger.info(f"Training Model for Target: {Config.TARGET_EAST}")
        y_train_east = train_df[Config.TARGET_EAST]
        y_val_east = val_df[Config.TARGET_EAST]

        dtrain_east = lgb.Dataset(X_train, label=y_train_east)
        dval_east = lgb.Dataset(X_val, label=y_val_east, reference=dtrain_east)

        callbacks_east = [
            lgb.early_stopping(stopping_rounds=Config.EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=Config.VERBOSE_EVAL),
        ]

        self.model_east = lgb.train(
            self.params,
            dtrain_east,
            num_boost_round=num_boost_round,
            valid_sets=[dtrain_east, dval_east],
            valid_names=["train", "valid"],
            callbacks=callbacks_east,
        )

        # Save East Model
        self.model_east.save_model(self.model_path_east)
        logger.info(f"East model saved to {self.model_path_east}")

        # ---------------------------------------------------------------------
        # Train North Model
        # ---------------------------------------------------------------------
        logger.info(f"Training Model for Target: {Config.TARGET_NORTH}")
        y_train_north = train_df[Config.TARGET_NORTH]
        y_val_north = val_df[Config.TARGET_NORTH]

        dtrain_north = lgb.Dataset(X_train, label=y_train_north)
        dval_north = lgb.Dataset(X_val, label=y_val_north, reference=dtrain_north)

        callbacks_north = [
            lgb.early_stopping(stopping_rounds=Config.EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=Config.VERBOSE_EVAL),
        ]

        self.model_north = lgb.train(
            self.params,
            dtrain_north,
            num_boost_round=num_boost_round,
            valid_sets=[dtrain_north, dval_north],
            valid_names=["train", "valid"],
            callbacks=callbacks_north,
        )

        # Save North Model
        self.model_north.save_model(self.model_path_north)
        logger.info(f"North model saved to {self.model_path_north}")

    def predict(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates predictions for the test set using the trained models.

        Args:
            test_df (pd.DataFrame): Test data containing features.

        Returns:
            pd.DataFrame: DataFrame containing 'delta_east' and 'delta_north' predictions.
        """
        # Load models if not in memory
        if self.model_east is None:
            if os.path.exists(self.model_path_east):
                logger.info(f"Loading East model from {self.model_path_east}")
                self.model_east = lgb.Booster(model_file=self.model_path_east)
            else:
                raise FileNotFoundError(
                    f"East model not found at {self.model_path_east}. Train first."
                )

        if self.model_north is None:
            if os.path.exists(self.model_path_north):
                logger.info(f"Loading North model from {self.model_path_north}")
                self.model_north = lgb.Booster(model_file=self.model_path_north)
            else:
                raise FileNotFoundError(
                    f"North model not found at {self.model_path_north}. Train first."
                )

        X_test = test_df[self.features]

        logger.info("Predicting East residuals...")
        pred_east = self.model_east.predict(
            X_test, num_iteration=self.model_east.best_iteration
        )

        logger.info("Predicting North residuals...")
        pred_north = self.model_north.predict(
            X_test, num_iteration=self.model_north.best_iteration
        )

        # Return predictions aligned with input index
        predictions = pd.DataFrame(index=test_df.index)
        predictions[Config.TARGET_EAST] = pred_east
        predictions[Config.TARGET_NORTH] = pred_north

        return predictions
