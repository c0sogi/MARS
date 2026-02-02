import os
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from library.config import Config
from library.utils import set_seed


class Stage1Ridge:
    """
    Stage 1: Sparse Lexical Regressor (Ridge).
    Uses TF-IDF features to predict normalized rank.
    Acts as the 'Signpost' model in the stacked architecture.
    """

    def __init__(self):
        self.model = None
        self.alpha = Config.RIDGE_ALPHA
        set_seed(Config.SEED)

    def fit(self, X, y):
        """
        Fits the Ridge model on the full dataset provided and saves the artifact.

        Args:
            X: Sparse TF-IDF matrix.
            y: Target values (normalized rank).
        """
        print(f"Training Stage 1 Ridge Model (alpha={self.alpha})...")
        self.model = Ridge(alpha=self.alpha, random_state=Config.SEED)
        self.model.fit(X, y)

        # Save model artifact
        os.makedirs(os.path.dirname(Config.RIDGE_MODEL_PATH), exist_ok=True)
        joblib.dump(self.model, Config.RIDGE_MODEL_PATH)
        print(f"Stage 1 model saved to {Config.RIDGE_MODEL_PATH}")

    def predict(self, X):
        """
        Predicts using the trained Ridge model. Loads from disk if necessary.

        Args:
            X: Sparse TF-IDF matrix.

        Returns:
            np.array: Predicted ranks.
        """
        if self.model is None:
            if os.path.exists(Config.RIDGE_MODEL_PATH):
                self.model = joblib.load(Config.RIDGE_MODEL_PATH)
            else:
                raise FileNotFoundError("Ridge model not found. Call fit() first.")

        return self.model.predict(X)

    def get_oof_predictions(self, X, y):
        """
        Generates Out-Of-Fold (OOF) predictions for the training set using K-Fold CV.
        These predictions serve as a bias-reduced feature for the Stage 2 model.

        Args:
            X: Sparse TF-IDF matrix.
            y: Target values.

        Returns:
            np.array: OOF predictions aligned with X.
        """
        print(f"Generating OOF predictions with {Config.NUM_FOLDS}-Fold CV...")
        kf = KFold(n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED)

        oof_preds = np.zeros(X.shape[0])
        # Ensure y is accessible by index
        y_np = y.values if isinstance(y, pd.Series) else y

        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_np)):
            # Split data for this fold
            X_train, X_val = X[train_idx], X[val_idx]
            y_train = y_np[train_idx]

            # Train temporary model
            model = Ridge(alpha=self.alpha, random_state=Config.SEED)
            model.fit(X_train, y_train)

            # Predict on validation fold
            oof_preds[val_idx] = model.predict(X_val)

        return oof_preds


class Stage2LGBM:
    """
    Stage 2: Dual-Anchor Gradient Booster (LightGBM).
    Refines predictions using Ridge output, Lexical/Semantic Anchor features, and LSA.
    """

    def __init__(self):
        self.model = None
        self.params = Config.get_lgbm_params()
        set_seed(Config.SEED)

    def fit(self, train_df, val_df, feature_cols, target_col):
        """
        Fits the LightGBM model using the provided training and validation sets.
        Implements Early Stopping to prevent overfitting.

        Args:
            train_df (pd.DataFrame): Training data.
            val_df (pd.DataFrame): Validation data.
            feature_cols (list): List of feature column names.
            target_col (str): Name of the target column.
        """
        print("Training Stage 2 LightGBM Model...")

        # Prepare Data
        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        X_val = val_df[feature_cols]
        y_val = val_df[target_col]

        train_set = lgb.Dataset(X_train, label=y_train)
        val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

        # Extract n_estimators to use as num_boost_round
        num_boost_round = self.params.pop("n_estimators", 2000)

        # Callbacks
        callbacks = [
            lgb.early_stopping(
                stopping_rounds=Config.EARLY_STOPPING_ROUNDS, verbose=False
            ),
            lgb.log_evaluation(period=100),
        ]

        # Train
        self.model = lgb.train(
            params=self.params,
            train_set=train_set,
            num_boost_round=num_boost_round,
            valid_sets=[train_set, val_set],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

        # Save model artifact
        os.makedirs(os.path.dirname(Config.LGBM_MODEL_PATH), exist_ok=True)
        self.model.save_model(Config.LGBM_MODEL_PATH)
        print(f"Stage 2 model saved to {Config.LGBM_MODEL_PATH}")

        # Calculate and print final validation metric
        y_pred_val = self.model.predict(X_val, num_iteration=self.model.best_iteration)
        mae = np.mean(np.abs(y_val - y_pred_val))
        print(f"Final Validation MAE: {mae}")

    def predict(self, df, feature_cols):
        """
        Predicts using the trained LightGBM model. Loads from disk if necessary.

        Args:
            df (pd.DataFrame): Dataframe containing features.
            feature_cols (list): List of feature columns to use.

        Returns:
            np.array: Predicted ranks.
        """
        if self.model is None:
            if os.path.exists(Config.LGBM_MODEL_PATH):
                self.model = lgb.Booster(model_file=Config.LGBM_MODEL_PATH)
            else:
                raise FileNotFoundError("LGBM model not found. Call fit() first.")

        # Predict using the best iteration determined during training
        return self.model.predict(
            df[feature_cols], num_iteration=self.model.best_iteration
        )
