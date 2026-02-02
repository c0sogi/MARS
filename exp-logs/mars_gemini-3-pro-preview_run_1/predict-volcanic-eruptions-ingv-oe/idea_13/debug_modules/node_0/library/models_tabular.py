import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from library.config import Config


class LightGBMTrainer:
    """
    Encapsulates the training and inference logic for the LightGBM Tabular Regressor.
    Implements Branch A of the Magnitude-Injected Hybrid Ensemble.
    """

    def __init__(self):
        """
        Initializes the trainer with parameters from Config.
        """
        self.params = Config.LGBM_PARAMS.copy()
        self.feature_cols = None

    def train(self, df_train, df_val, fold_id=0):
        """
        Trains a LightGBM model on the provided training and validation DataFrames.
        Implements early stopping based on Validation MAE.

        Args:
            df_train (pd.DataFrame): Training data containing features and target.
            df_val (pd.DataFrame): Validation data containing features and target.
            fold_id (int): The fold index, used for saving the model file.

        Returns:
            lgb.Booster: The trained LightGBM model.
        """
        # Identify feature columns: Exclude metadata and target
        ignore_cols = {"segment_id", "time_to_eruption", "file_path"}
        self.feature_cols = [c for c in df_train.columns if c not in ignore_cols]

        # Prepare X and y
        X_train = df_train[self.feature_cols]
        y_train = df_train["time_to_eruption"]

        X_val = df_val[self.feature_cols]
        y_val = df_val["time_to_eruption"]

        # Create LightGBM Datasets
        # We train on raw targets as tree models are scale-invariant regarding features
        # and can handle large target ranges.
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        # Setup Callbacks
        callbacks = [
            lgb.early_stopping(
                stopping_rounds=self.params.get("early_stopping_rounds", 100),
                verbose=True,
            ),
            lgb.log_evaluation(period=100),
        ]

        # Train the model
        print(f"Starting training for Fold {fold_id}...")
        evals_result = {}
        model = lgb.train(
            self.params,
            dtrain,
            num_boost_round=self.params.get("n_estimators", 5000),
            valid_sets=[dtrain, dval],
            valid_names=["train", "valid"],
            callbacks=callbacks,
            evals_result=evals_result,
        )

        # Save the model
        model_filename = f"lgb_model_fold_{fold_id}.txt"
        model_path = os.path.join(Config.IDEA_DIR, model_filename)
        model.save_model(model_path)
        print(f"Model for Fold {fold_id} saved to {model_path}")

        # Report Best Score
        # Accessing the best score from the valid set
        if "valid" in model.best_score and "mae" in model.best_score["valid"]:
            best_mae = model.best_score["valid"]["mae"]
            print(f"Fold {fold_id} Best Validation MAE: {best_mae}")

        return model

    def predict(self, df, model=None, fold_id=None):
        """
        Generates predictions for a given DataFrame.

        Args:
            df (pd.DataFrame): Data to predict on.
            model (lgb.Booster, optional): A trained model instance.
            fold_id (int, optional): If model is None, load model for this fold.

        Returns:
            np.ndarray: Array of predicted values.
        """
        # Resolve Model
        if model is None:
            if fold_id is not None:
                model = self.load_model(fold_id)
            else:
                raise ValueError("Must provide either a model instance or a fold_id.")

        # Resolve Features
        # If self.feature_cols is not set (e.g. fresh instance), infer from df
        if self.feature_cols is None:
            ignore_cols = {"segment_id", "time_to_eruption", "file_path"}
            features = [c for c in df.columns if c not in ignore_cols]
        else:
            features = self.feature_cols

        # Validate Features
        missing_features = set(features) - set(df.columns)
        if missing_features:
            raise ValueError(f"Input DataFrame is missing features: {missing_features}")

        X = df[features]

        # Predict
        predictions = model.predict(X)
        return predictions

    def load_model(self, fold_id):
        """
        Loads a saved LightGBM model from the working directory.

        Args:
            fold_id (int): The fold index of the model to load.

        Returns:
            lgb.Booster: The loaded model.
        """
        model_filename = f"lgb_model_fold_{fold_id}.txt"
        model_path = os.path.join(Config.IDEA_DIR, model_filename)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"LightGBM model file not found at {model_path}")

        model = lgb.Booster(model_file=model_path)
        return model
