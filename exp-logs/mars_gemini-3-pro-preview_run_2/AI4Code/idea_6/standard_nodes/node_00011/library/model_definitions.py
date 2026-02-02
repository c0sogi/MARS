import os
import joblib
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from library.config import Config


class Level1Ridge:
    """
    Wrapper for the Level 1 Ridge Regression model.
    Predicts cell rank based on sparse TF-IDF features.
    """

    def __init__(self):
        self.model = Ridge(
            alpha=Config.RIDGE_ALPHA,
            fit_intercept=True,
            solver="auto",
            random_state=Config.SEED,
        )

    def fit(self, X, y):
        """
        Trains the Ridge model.
        """
        print("Training Level 1 Ridge Model...")
        self.model.fit(X, y)

    def predict(self, X):
        """
        Generates predictions.
        """
        return self.model.predict(X)

    def save(self, path):
        """
        Saves the model to disk using joblib.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"Level 1 Ridge model saved to {path}")

    def load(self, path):
        """
        Loads the model from disk.
        """
        if os.path.exists(path):
            self.model = joblib.load(path)
            print(f"Level 1 Ridge model loaded from {path}")
        else:
            raise FileNotFoundError(f"Model file not found at {path}")


class Level2GBM:
    """
    Wrapper for the Level 2 LightGBM Regressor.
    Refines predictions using stacked features and metadata.
    """

    def __init__(self):
        # Copy params to avoid mutating the global Config
        params = Config.LGBM_PARAMS.copy()

        # Extract early_stopping_rounds to use in callbacks
        self.es_rounds = params.pop("early_stopping_rounds", None)

        self.model = LGBMRegressor(**params)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains the LightGBM model with optional early stopping.
        """
        print("Training Level 2 LightGBM Model...")

        callbacks = []
        eval_set = None

        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]

            # Configure callbacks
            # 1. Log evaluation (period=0 suppresses intermediate output)
            callbacks.append(log_evaluation(period=0))

            # 2. Early stopping
            if self.es_rounds:
                callbacks.append(
                    early_stopping(stopping_rounds=self.es_rounds, verbose=False)
                )

        self.model.fit(
            X_train, y_train, eval_set=eval_set, eval_metric="mae", callbacks=callbacks
        )

        # Print the best validation score if validation data was provided
        if eval_set and self.model.best_score_:
            # 'valid_0' is the default name for the first eval set
            # 'l1' corresponds to 'mae' in LightGBM
            metric_name = "l1"
            if (
                "valid_0" in self.model.best_score_
                and metric_name in self.model.best_score_["valid_0"]
            ):
                score = self.model.best_score_["valid_0"][metric_name]
                print(f"Best Validation MAE: {score}")

    def predict(self, X):
        """
        Generates predictions.
        """
        return self.model.predict(X)

    def save(self, path):
        """
        Saves the model to disk using joblib.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"Level 2 LightGBM model saved to {path}")

    def load(self, path):
        """
        Loads the model from disk.
        """
        if os.path.exists(path):
            self.model = joblib.load(path)
            print(f"Level 2 LightGBM model loaded from {path}")
        else:
            raise FileNotFoundError(f"Model file not found at {path}")
