import numpy as np
import xgboost as xgb
import os
import joblib
from library.config import Config
from library.utils import setup_seed


class ContactModel:
    """
    Wrapper class for XGBoost model with specific handling for:
    - Targeted Majority Undersampling
    - GPU acceleration
    - Asymmetric parameter configuration (Stream A vs Stream B)
    """

    def __init__(self, params):
        """
        Initialize the model with specific hyperparameters.

        Args:
            params (dict): Dictionary containing XGBoost parameters.
                           Must include 'n_estimators' and 'early_stopping_rounds'.
        """
        self.params = params.copy()

        # Extract training control parameters that are not part of the booster config
        self.n_estimators = self.params.pop("n_estimators", 1000)
        self.early_stopping_rounds = self.params.pop("early_stopping_rounds", 50)

        # Ensure thread count is set
        if "nthread" not in self.params:
            self.params["nthread"] = Config.N_JOBS

        self.model = None
        self.best_iteration = None

    def _undersample(self, X, y):
        """
        Performs Targeted Majority Undersampling.
        Retains 100% of positives and subsamples negatives based on Config.UNDERSAMPLE_RATIO.
        """
        # Identify indices
        pos_indices = np.where(y == 1)[0]
        neg_indices = np.where(y == 0)[0]

        n_pos = len(pos_indices)
        n_neg = len(neg_indices)

        # Calculate target number of negatives
        target_neg = int(n_pos * Config.UNDERSAMPLE_RATIO)

        if target_neg >= n_neg:
            # If we want more negatives than exist (or equal), keep all
            return X, y

        # Randomly sample negatives
        # Use Config.SEED for reproducibility within the sampling
        rng = np.random.RandomState(Config.SEED)
        sampled_neg_indices = rng.choice(neg_indices, size=target_neg, replace=False)

        # Combine and shuffle
        final_indices = np.concatenate([pos_indices, sampled_neg_indices])
        rng.shuffle(final_indices)

        if isinstance(X, np.ndarray):
            return X[final_indices], y[final_indices]
        else:
            # Handle pandas DataFrame if passed (though pipeline uses numpy)
            return X.iloc[final_indices], y[final_indices]

    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=False):
        """
        Trains the XGBoost model.

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features (optional).
            y_val: Validation labels (optional).
            verbose: Whether to print training progress.
        """
        setup_seed(Config.SEED)

        # Apply Undersampling to Training Data
        X_train_res, y_train_res = self._undersample(X_train, y_train)

        # Create DMatrices
        dtrain = xgb.DMatrix(X_train_res, label=y_train_res)

        evals = [(dtrain, "train")]
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, "validation"))

        # Train
        self.model = xgb.train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=self.n_estimators,
            evals=evals,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=100 if verbose else False,
        )

        self.best_iteration = self.model.best_iteration

        if verbose:
            print(f"Best iteration: {self.best_iteration}")
            print(f"Best score: {self.model.best_score}")

    def predict(self, X):
        """
        Generates probability predictions.

        Args:
            X: Features.

        Returns:
            np.ndarray: Probability of class 1.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")

        dtest = xgb.DMatrix(X)
        # Predict using the best iteration found during training
        return self.model.predict(dtest, iteration_range=(0, self.best_iteration + 1))

    def save(self, filepath):
        """
        Saves the model object to disk.
        """
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath):
        """
        Loads a model object from disk.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        return joblib.load(filepath)
