import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from library.config import RF_PARAMS


class PizzaRandomForest:
    """
    A wrapper class for the Random Forest Classifier to predict pizza request success.
    """

    def __init__(self):
        """
        Initialize the Random Forest model with parameters from config.
        """
        self.model = RandomForestClassifier(**RF_PARAMS)

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains the Random Forest model and evaluates on validation set if provided.

        Args:
            X_train (pd.DataFrame or np.ndarray): Training features.
            y_train (pd.Series or np.ndarray): Training labels.
            X_val (pd.DataFrame or np.ndarray, optional): Validation features.
            y_val (pd.Series or np.ndarray, optional): Validation labels.
        """
        print("Starting training of Random Forest Classifier...")
        self.model.fit(X_train, y_train)

        # Calculate and print Training AUC
        train_preds = self.predict_proba(X_train)
        train_auc = roc_auc_score(y_train, train_preds)
        print(f"Training ROC AUC: {train_auc}")

        # Calculate and print Validation AUC if validation data is available
        if X_val is not None and y_val is not None:
            val_preds = self.predict_proba(X_val)
            val_auc = roc_auc_score(y_val, val_preds)
            print(f"Validation ROC AUC: {val_auc}")

    def predict_proba(self, X):
        """
        Generates probability predictions for the positive class (1).

        Args:
            X (pd.DataFrame or np.ndarray): Features to predict on.

        Returns:
            np.ndarray: Probabilities of the positive class.
        """
        # predict_proba returns an array of shape (n_samples, n_classes)
        # We take the second column (index 1) for the probability of class 1.
        return self.model.predict_proba(X)[:, 1]
