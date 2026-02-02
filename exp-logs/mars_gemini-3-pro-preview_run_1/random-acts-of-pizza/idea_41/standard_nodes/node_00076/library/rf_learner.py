import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from library.config import RF_PARAMS
from library.utils import set_seed


class RFPredictor:
    """
    Random Forest Predictor for the pizza request success prediction task.
    Encapsulates the training and prediction logic using scikit-learn's RandomForestClassifier.
    """

    def __init__(self, params=None):
        """
        Initialize the RFPredictor.

        Args:
            params (dict, optional): Hyperparameters for the RandomForestClassifier.
                                     If None, defaults to RF_PARAMS from config.
        """
        self.params = params if params is not None else RF_PARAMS.copy()
        self.model = None

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains the Random Forest model.

        Args:
            X_train (sparse matrix or array-like): Training features.
            y_train (array-like): Training targets.
            X_val (sparse matrix or array-like, optional): Validation features.
            y_val (array-like, optional): Validation targets.
        """
        # Ensure reproducibility
        set_seed()

        print("Initializing Random Forest Classifier...")
        self.model = RandomForestClassifier(**self.params)

        print("Fitting Random Forest model...")
        self.model.fit(X_train, y_train)

        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            print("Evaluating on validation set...")
            val_probs = self.predict(X_val)
            val_auc = roc_auc_score(y_val, val_probs)
            # Print full precision as requested
            print(f"Random Forest Validation AUC: {val_auc}")

    def predict(self, X):
        """
        Generates predictions for the given input features.

        Args:
            X (sparse matrix or array-like): Features to predict on.

        Returns:
            np.ndarray: Predicted probabilities for the positive class (class 1).
        """
        if self.model is None:
            raise RuntimeError(
                "The model has not been trained yet. Call train() first."
            )

        # predict_proba returns [n_samples, n_classes], we want the probability of class 1
        return self.model.predict_proba(X)[:, 1]
