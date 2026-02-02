import xgboost as xgb
import numpy as np
from library import config


class XGBTrainer:
    """
    Wrapper class for XGBoost training and inference, optimized for
    self-training ensembles with probability outputs.
    """

    def __init__(self, params=None):
        """
        Initialize the trainer with XGBoost parameters.

        Args:
            params (dict, optional): XGBoost parameters. Defaults to config.XGB_PARAMS.
        """
        # Create a copy to avoid modifying the global config dictionary
        self.params = params.copy() if params else config.XGB_PARAMS.copy()

        # Force objective to 'multi:softprob' to ensure the model outputs probabilities.
        # The config specifies 'multi:softmax' (class labels), but the ensemble strategy
        # requires Soft Voting (averaging probabilities) and LogLoss optimization.
        if self.params.get("objective") == "multi:softmax":
            self.params["objective"] = "multi:softprob"

        self.model = None

    def train(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        num_boost_round=None,
        early_stopping_rounds=None,
        verbose_eval=False,
    ):
        """
        Train the XGBoost model with Early Stopping.

        Args:
            X_train (pd.DataFrame or np.ndarray): Training features.
            y_train (pd.Series or np.ndarray): Training targets.
            X_val (pd.DataFrame or np.ndarray, optional): Validation features.
            y_val (pd.Series or np.ndarray, optional): Validation targets.
            num_boost_round (int, optional): Maximum number of boosting rounds.
            early_stopping_rounds (int, optional): Rounds for early stopping.
            verbose_eval (bool or int, optional): Logging verbosity.
        """
        if num_boost_round is None:
            num_boost_round = config.NUM_BOOST_ROUND
        if early_stopping_rounds is None:
            early_stopping_rounds = config.EARLY_STOPPING_ROUNDS

        # Create DMatrix for efficient training
        dtrain = xgb.DMatrix(X_train, label=y_train)

        evals = [(dtrain, "train")]
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, "val"))

        # Train the model
        self.model = xgb.train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval,
        )

        # Output best performance metrics
        if X_val is not None and y_val is not None:
            print(f"Best Iteration: {self.model.best_iteration}")
            # Print full precision as requested
            print(f"Best Score: {self.model.best_score}")

    def predict(self, X):
        """
        Generate probability predictions using the trained model.

        Args:
            X (pd.DataFrame or np.ndarray): Input features.

        Returns:
            np.ndarray: Predicted probabilities with shape (n_samples, n_classes).
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)
