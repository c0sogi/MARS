import os
import joblib
import pandas as pd
import lightgbm as lgb
from library.config import Config


class LGBMRanker:
    """
    Implements the Stage 2 Multi-Resolution Gradient Booster using LightGBM.

    Responsibilities:
    1. Train a LightGBM Regressor on the combined feature set (Ridge OOF + Anchors + Metadata).
    2. Implement Early Stopping using a validation set.
    3. Manage caching of the trained model.
    4. Predict normalized ranks for test data.
    """

    def __init__(self):
        self.config = Config
        self.params = self.config.LGBM_PARAMS
        self.early_stopping_rounds = self.config.EARLY_STOPPING_ROUNDS
        self.working_dir = self.config.WORKING_DIR

        # Placeholder for the model
        self.model = None
        self.is_fitted = False

    def _get_model_path(self):
        """Returns the path for caching the trained model."""
        return os.path.join(self.working_dir, "stage2_lgbm_model.joblib")

    def fit(self, X_train, y_train, X_val=None, y_val=None, load_cached_model=True):
        """
        Fits the LightGBM model. If a cached model exists and load_cached_model is True,
        loads it instead of retraining.

        Args:
            X_train (pd.DataFrame or np.array): Training features.
            y_train (pd.Series or np.array): Training targets (normalized ranks).
            X_val (pd.DataFrame or np.array, optional): Validation features.
            y_val (pd.Series or np.array, optional): Validation targets.
            load_cached_model (bool): Whether to attempt loading from cache.
        """
        model_path = self._get_model_path()

        # 1. Try to load from cache
        if load_cached_model and os.path.exists(model_path):
            print(f"Loading Stage 2 LightGBM model from {model_path}")
            try:
                self.model = joblib.load(model_path)
                self.is_fitted = True
                return
            except Exception as e:
                print(f"Failed to load cache: {e}. Reprocessing...")

        print("Starting Stage 2 LightGBM training...")

        # 2. Initialize Model
        # Note: We use the sklearn API wrapper provided by LightGBM
        self.model = lgb.LGBMRegressor(**self.params)

        # 3. Setup Callbacks and Eval Set
        callbacks = []
        eval_set = None

        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            # Add early stopping callback
            callbacks.append(
                lgb.early_stopping(
                    stopping_rounds=self.early_stopping_rounds, verbose=True
                )
            )
            # Add logging callback to print metrics
            callbacks.append(lgb.log_evaluation(period=100))

        # 4. Train
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            eval_metric=self.params.get("metric", "mae"),
            callbacks=callbacks,
        )

        self.is_fitted = True

        # 5. Save to Cache
        print(f"Saving Stage 2 LightGBM model to {model_path}")
        joblib.dump(self.model, model_path)

    def predict(self, X):
        """
        Predicts normalized ranks using the fitted model.

        Args:
            X (pd.DataFrame or np.array): Feature matrix.

        Returns:
            np.array: Predicted normalized ranks.
        """
        if not self.is_fitted:
            # Attempt to load if not in memory
            model_path = self._get_model_path()
            if os.path.exists(model_path):
                print(
                    f"Loading Stage 2 LightGBM model from {model_path} for inference..."
                )
                self.model = joblib.load(model_path)
                self.is_fitted = True
            else:
                raise RuntimeError("Model is not fitted and no cached model found.")

        # Predict
        return self.model.predict(X)
