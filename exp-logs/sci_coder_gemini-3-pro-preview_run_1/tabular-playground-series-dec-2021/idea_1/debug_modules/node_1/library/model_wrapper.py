import sys
import xgboost as xgb
import numpy as np

# Add library path to allow imports from utils.py
sys.path.insert(0, "./library")
from utils import print_metrics


class XGBClassifierWrapper:
    """
    A wrapper class for XGBoost training and inference.
    Encapsulates model configuration, training with early stopping, and prediction.
    """

    def __init__(
        self,
        params=None,
        num_boost_round=3000,
        early_stopping_rounds=50,
        verbose_eval=50,
    ):
        """
        Initialize the XGBClassifierWrapper.

        Args:
            params (dict, optional): XGBoost hyperparameters.
            num_boost_round (int): Maximum number of boosting rounds.
            early_stopping_rounds (int): Rounds of no improvement to trigger early stopping.
            verbose_eval (int): Frequency of printing evaluation metrics.
        """
        # Default parameters for GPU-accelerated multi-class classification
        # Note: 'tree_method': 'hist' with 'device': 'cuda' is the modern equivalent
        # of 'gpu_hist' for XGBoost versions >= 2.0 to utilize the A100 GPU.
        self.default_params = {
            "objective": "multi:softmax",
            "num_class": 6,  # Targets are encoded 0-5
            "tree_method": "hist",
            "device": "cuda",
            "eval_metric": ["merror"],
            "eta": 0.1,
            "max_depth": 8,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "verbosity": 1,
            "seed": 42,
        }

        # Update defaults with provided params
        self.params = self.default_params.copy()
        if params:
            self.params.update(params)

        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose_eval = verbose_eval
        self.model = None

    def train(self, dtrain, dval):
        """
        Trains the XGBoost model using the provided training and validation sets.
        Implements early stopping based on validation performance.

        Args:
            dtrain (xgb.DMatrix): Training data.
            dval (xgb.DMatrix): Validation data.
        """
        watchlist = [(dtrain, "train"), (dval, "val")]

        print(f"Training XGBoost model with params: {self.params}")

        self.model = xgb.train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=self.num_boost_round,
            evals=watchlist,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=self.verbose_eval,
        )

        # Report best performance
        if self.model.best_score is not None:
            # Determine metric name for logging
            metric_list = self.params.get("eval_metric", ["error"])
            metric_name = (
                metric_list[-1] if isinstance(metric_list, list) else metric_list
            )

            print_metrics(
                {
                    f"best_{metric_name}": self.model.best_score,
                    "best_iteration": self.model.best_iteration,
                }
            )

    def predict(self, dtest):
        """
        Generates predictions for the test set.

        Args:
            dtest (xgb.DMatrix): Test data.

        Returns:
            np.array: Predicted class indices.
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call train() first.")

        # Predict class indices
        # For multi:softmax, output is class label (int)
        predictions = self.model.predict(dtest)
        return predictions
