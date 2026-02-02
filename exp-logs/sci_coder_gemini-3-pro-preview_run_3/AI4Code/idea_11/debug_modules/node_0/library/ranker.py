import os
import lightgbm as lgb
import pandas as pd
import numpy as np
from library.config import Config, set_seed


class LGBMRanker:
    """
    Encapsulates the LightGBM Regressor for the Multi-Scale Structural Heatmap Regressor pipeline.
    Trains on extracted features to predict the normalized position of markdown cells.
    """

    def __init__(self):
        self.model = None
        # Columns to exclude from features
        self.ignore_cols = {"id", "cell_id", "target"}

    def train(self, train_df, val_df):
        """
        Trains the LightGBM model using the provided training and validation dataframes.

        Args:
            train_df (pd.DataFrame): Training features and targets.
            val_df (pd.DataFrame): Validation features and targets.
        """
        set_seed(Config.SEED)

        # 1. Feature Selection
        # dynamically select all columns that are not metadata or target
        feature_cols = [c for c in train_df.columns if c not in self.ignore_cols]
        print(f"Training LightGBM with {len(feature_cols)} features.")

        # 2. Prepare LightGBM Datasets
        X_train = train_df[feature_cols]
        y_train = train_df["target"]
        X_val = val_df[feature_cols]
        y_val = val_df["target"]

        train_data = lgb.Dataset(X_train, label=y_train)
        # Reference train_data ensures bin alignment for validation
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # 3. Setup Parameters
        params = Config.get_lgbm_params()
        # Extract n_estimators to use as num_boost_round argument
        num_boost_round = params.pop("n_estimators", 1000)

        # 4. Callbacks
        callbacks = [
            lgb.early_stopping(
                stopping_rounds=Config.EARLY_STOPPING_ROUNDS, verbose=True
            ),
            lgb.log_evaluation(period=50),
        ]

        # 5. Train
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[train_data, val_data],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

        # 6. Save Model
        os.makedirs(os.path.dirname(Config.LGBM_MODEL_PATH), exist_ok=True)
        self.model.save_model(Config.LGBM_MODEL_PATH)
        print(f"Model saved to {Config.LGBM_MODEL_PATH}")

        # 7. Print Metrics (Full Precision)
        if self.model.best_score:
            print("-" * 30)
            print("Best Validation Score (Full Precision):")
            for dataset_name, metrics in self.model.best_score.items():
                for metric_name, score in metrics.items():
                    print(f"{dataset_name} {metric_name}: {score}")
            print("-" * 30)

    def predict(self, features_df):
        """
        Generates predictions for the given features.

        Args:
            features_df (pd.DataFrame): DataFrame containing features.

        Returns:
            np.ndarray: Predicted normalized ranks.
        """
        # 1. Load Model if needed
        if self.model is None:
            if os.path.exists(Config.LGBM_MODEL_PATH):
                # Load model with silent mode to avoid extra prints
                self.model = lgb.Booster(model_file=Config.LGBM_MODEL_PATH)
            else:
                raise RuntimeError(
                    f"Model not found at {Config.LGBM_MODEL_PATH}. Please train first."
                )

        # 2. Feature Selection
        # Ensure we use the exact same logic as training
        feature_cols = [c for c in features_df.columns if c not in self.ignore_cols]

        # 3. Predict
        return self.model.predict(features_df[feature_cols])
