import os
import joblib
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import Ridge
from library import config, utils


# ------------------------------------------------------------------------------
# Stage 1: Sparse Lexical Regressor (Ridge)
# ------------------------------------------------------------------------------
class Stage1Ridge:
    def __init__(self, params=None):
        """
        Initializes the Stage 1 Ridge Regressor.
        Args:
            params (dict, optional): Hyperparameters for Ridge. Defaults to config.RIDGE_PARAMS.
        """
        self.params = params if params is not None else config.RIDGE_PARAMS
        self.model = Ridge(**self.params)
        self.model_path = config.CACHE_RIDGE_MODEL

    def fit(self, X, y):
        """
        Fits the Ridge model on sparse features.
        Args:
            X (scipy.sparse.csr_matrix): Sparse TF-IDF matrix.
            y (array-like): Target normalized ranks.
        """
        utils.log_message("Training Stage 1 (Ridge) Model...")
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """
        Predicts normalized ranks using the Ridge model.
        Args:
            X (scipy.sparse.csr_matrix): Sparse TF-IDF matrix.
        Returns:
            np.ndarray: Predicted ranks.
        """
        return self.model.predict(X)

    def save(self):
        """Saves the fitted model to disk using joblib."""
        utils.log_message(f"Saving Stage 1 model to {self.model_path}...")
        joblib.dump(self.model, self.model_path)

    def load(self):
        """Loads the fitted model from disk."""
        if os.path.exists(self.model_path):
            utils.log_message(f"Loading Stage 1 model from {self.model_path}...")
            self.model = joblib.load(self.model_path)
        else:
            raise FileNotFoundError(f"Stage 1 model not found at {self.model_path}")
        return self


# ------------------------------------------------------------------------------
# Stage 2: Multi-View Neighborhood Gradient Booster (LightGBM)
# ------------------------------------------------------------------------------
class Stage2LGBM:
    def __init__(self, params=None):
        """
        Initializes the Stage 2 LightGBM Regressor.
        Args:
            params (dict, optional): Hyperparameters for LGBM. Defaults to config.LGBM_PARAMS.
        """
        self.params = params if params is not None else config.LGBM_PARAMS
        self.model = None
        self.model_path = os.path.join(config.WORKING_DIR, "lgbm_model.txt")

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Fits the LightGBM model with Early Stopping.
        Args:
            X_train (pd.DataFrame or np.ndarray): Training features.
            y_train (pd.Series or np.ndarray): Training targets.
            X_val (pd.DataFrame or np.ndarray, optional): Validation features.
            y_val (pd.Series or np.ndarray, optional): Validation targets.
        """
        utils.log_message("Initializing Stage 2 (LightGBM) Model...")

        # Initialize the Scikit-Learn API wrapper
        self.model = lgb.LGBMRegressor(**self.params)

        callbacks = []
        eval_set = None

        # Configure validation and callbacks if validation data is provided
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            callbacks.append(
                lgb.early_stopping(stopping_rounds=config.EARLY_STOPPING_ROUNDS)
            )
            callbacks.append(lgb.log_evaluation(period=config.VERBOSE_EVAL))
            utils.log_message("Training Stage 2 with Early Stopping...")
        else:
            utils.log_message("Training Stage 2 without Validation...")

        # Train the model
        self.model.fit(
            X_train, y_train, eval_set=eval_set, eval_metric="mae", callbacks=callbacks
        )

        if X_val is not None and y_val is not None:
            # Retrieve and log the best score from the validation set
            # 'valid_0' is the default name for the first eval_set
            best_score = self.model.best_score_["valid_0"]["l1"]
            utils.log_message(f"Best Validation MAE: {best_score}")

        return self

    def predict(self, X):
        """
        Predicts normalized ranks using the LightGBM model.
        Args:
            X (pd.DataFrame or np.ndarray): Features.
        Returns:
            np.ndarray: Predicted ranks.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted.")
        return self.model.predict(X)

    def save(self):
        """Saves the LightGBM booster to a text file."""
        utils.log_message(f"Saving Stage 2 model to {self.model_path}...")
        if self.model is not None:
            # Save the underlying booster to text format (portable and efficient)
            self.model.booster_.save_model(self.model_path)
        else:
            raise ValueError("Cannot save empty model.")

    def load(self):
        """Loads the LightGBM booster from a text file into the Sklearn wrapper."""
        if os.path.exists(self.model_path):
            utils.log_message(f"Loading Stage 2 model from {self.model_path}...")

            # Re-initialize the wrapper with the same parameters
            self.model = lgb.LGBMRegressor(**self.params)

            # Load the booster from the text file
            booster = lgb.Booster(model_file=self.model_path)

            # Inject the booster into the wrapper
            self.model._Booster = booster
            self.model.fitted_ = True
        else:
            raise FileNotFoundError(f"Stage 2 model not found at {self.model_path}")
        return self
