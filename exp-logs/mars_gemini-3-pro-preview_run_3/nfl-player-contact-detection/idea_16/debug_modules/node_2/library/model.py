import os
import numpy as np
import xgboost as xgb
from library.utils import setup_seed


class ContactXGB:
    """
    Wrapper class for XGBoost Classifier to handle training, prediction,
    and persistence for the NFL Player Contact Detection task.
    """

    def __init__(self, params):
        """
        Initialize the XGBoost model with given parameters.

        Args:
            params (dict): Dictionary of hyperparameters.
                           'n_estimators' is extracted for the constructor.
        """
        self.params = params.copy()

        # Extract n_estimators as it is a constructor argument for XGBClassifier
        self.n_estimators = self.params.pop("n_estimators", 1000)

        # Initialize the XGBClassifier
        # random_state is passed via params (from Config)
        self.model = xgb.XGBClassifier(n_estimators=self.n_estimators, **self.params)

    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose_eval=50):
        """
        Train the model with Early Stopping.

        Args:
            X_train (pd.DataFrame or np.ndarray): Training features.
            y_train (np.ndarray): Training labels.
            X_val (pd.DataFrame or np.ndarray, optional): Validation features.
            y_val (np.ndarray, optional): Validation labels.
            verbose_eval (int): Period for printing training progress.
        """
        eval_set = []

        # If validation data is provided, add it to eval_set
        # We also include training set to monitor overfitting
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]

        print(f"Starting training with {self.n_estimators} estimators...")

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=verbose_eval,
        )

        # Print the best score with full precision
        if hasattr(self.model, "best_score"):
            print(f"Training finished. Best validation score: {self.model.best_score}")
        if hasattr(self.model, "best_iteration"):
            print(f"Best iteration: {self.model.best_iteration}")

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Args:
            X (pd.DataFrame or np.ndarray): Features.

        Returns:
            np.ndarray: Probabilities for the positive class (class 1).
        """
        # predict_proba returns [prob_0, prob_1]
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X):
        """
        Predict class labels (binary).
        Note: For this task, threshold optimization is usually performed on probabilities.

        Args:
            X (pd.DataFrame or np.ndarray): Features.

        Returns:
            np.ndarray: Binary predictions.
        """
        return self.model.predict(X)

    def save(self, path):
        """
        Save the model to a file using JSON format.

        Args:
            path (str): File path to save the model.
        """
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        # Ensure correct extension for XGBoost save_model
        if not path.endswith(".json"):
            path += ".json"

        print(f"Saving model to {path}...")
        self.model.save_model(path)

    def load(self, path):
        """
        Load the model from a file.

        Args:
            path (str): File path to load the model from.
        """
        if not path.endswith(".json"):
            # Try appending json if file doesn't exist exactly as named
            if not os.path.exists(path) and os.path.exists(path + ".json"):
                path += ".json"

        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")

        print(f"Loading model from {path}...")
        self.model.load_model(path)
