import os
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from library.config import Config


class Stage1Ridge:
    """
    Stage 1: Sparse Lexical Regressor.
    Wraps a Ridge Regression model optimized for high-dimensional sparse TF-IDF inputs.
    """

    def __init__(self):
        self.model_path = os.path.join(Config.WORKING_DIR, "stage1_ridge.joblib")
        self.model = Ridge(
            alpha=Config.RIDGE_ALPHA, random_state=Config.SEED, solver="auto"
        )
        self.is_fitted = False

    def fit(self, X, y):
        """
        Fits the Ridge model on sparse features.

        Args:
            X (scipy.sparse.csr_matrix): Sparse TF-IDF features.
            y (np.array): Target normalized ranks.
        """
        print(f"Training Stage 1 Ridge (Input shape: {X.shape})...")
        self.model.fit(X, y)
        self.is_fitted = True

        # Save model
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        joblib.dump(self.model, self.model_path)
        print(f"Stage 1 model saved to {self.model_path}")

    def predict(self, X):
        """
        Generates predictions using the Ridge model.
        Loads from cache if not currently fitted in memory.

        Args:
            X (scipy.sparse.csr_matrix): Sparse TF-IDF features.

        Returns:
            np.array: Predicted ranks.
        """
        if not self.is_fitted:
            if os.path.exists(self.model_path):
                print(f"Loading Stage 1 model from {self.model_path}...")
                self.model = joblib.load(self.model_path)
                self.is_fitted = True
            else:
                raise ValueError("Stage 1 model is not fitted and no cache found.")

        return self.model.predict(X)


class Stage2LGBM:
    """
    Stage 2: Multi-View Instance Gradient Booster.
    Wraps a LightGBM Regressor to refine predictions using dense instance-based features.
    """

    def __init__(self):
        self.model_path = os.path.join(Config.WORKING_DIR, "stage2_lgbm.txt")
        self.model = lgb.LGBMRegressor(**Config.LGBM_PARAMS)
        self.is_fitted = False

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Fits the LightGBM model with Early Stopping.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (np.array): Training targets.
            X_val (pd.DataFrame, optional): Validation features.
            y_val (np.array, optional): Validation targets.
        """
        print(f"Training Stage 2 LightGBM (Input shape: {X_train.shape})...")

        eval_set = []
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        # Configure callbacks for LightGBM 4.x
        callbacks = [
            lgb.log_evaluation(period=Config.LGBM_VERBOSE_EVAL),
        ]

        if eval_set:
            callbacks.append(
                lgb.early_stopping(stopping_rounds=Config.LGBM_EARLY_STOPPING_ROUNDS)
            )

        self.model.fit(
            X_train, y_train, eval_set=eval_set, eval_metric="mae", callbacks=callbacks
        )
        self.is_fitted = True

        # Save model (using LightGBM's native text format for portability/inspection)
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        self.model.booster_.save_model(self.model_path)
        print(f"Stage 2 model saved to {self.model_path}")

        # Print final validation metric if available
        if X_val is not None and y_val is not None:
            preds = self.model.predict(X_val)
            mae = mean_absolute_error(y_val, preds)
            print(f"Final Validation MAE: {mae}")

    def predict(self, X):
        """
        Generates predictions using the LightGBM model.
        Loads from cache if not currently fitted in memory.

        Args:
            X (pd.DataFrame): Features.

        Returns:
            np.array: Predicted ranks.
        """
        if not self.is_fitted:
            if os.path.exists(self.model_path):
                print(f"Loading Stage 2 model from {self.model_path}...")
                # Load into a Booster and wrap it back into the sklearn API container if needed,
                # or simply use the Booster to predict.
                # For consistency with sklearn API, we re-initialize and set the booster.
                booster = lgb.Booster(model_file=self.model_path)
                # We can use the booster directly for prediction
                return booster.predict(X)
            else:
                raise ValueError("Stage 2 model is not fitted and no cache found.")

        return self.model.predict(X)
