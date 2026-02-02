import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from library.config import Config


class PizzaRandomForest:
    """
    Wrapper for a Random Forest Classifier tailored for the Pizza Request dataset.
    Encapsulates training, prediction, and evaluation logic using configuration settings.
    """

    def __init__(self):
        """
        Initialize the Random Forest model with hyperparameters from Config.
        """
        self.config = Config()

        # Initialize the scikit-learn RandomForestClassifier
        self.model = RandomForestClassifier(
            n_estimators=self.config.RF_N_ESTIMATORS,
            max_depth=self.config.RF_MAX_DEPTH,
            max_features=self.config.RF_MAX_FEATURES,
            class_weight=self.config.RF_CLASS_WEIGHT,
            n_jobs=self.config.RF_N_JOBS,
            random_state=self.config.RANDOM_SEED,
            verbose=0,  # Keep sklearn verbose low, we print our own metrics
        )

        # Placeholder for feature names to map importances later
        self.feature_names = None

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains the Random Forest model.

        Args:
            X_train (pd.DataFrame or np.ndarray): Training features.
            y_train (pd.Series or np.ndarray): Training targets.
            X_val (pd.DataFrame or np.ndarray, optional): Validation features.
            y_val (pd.Series or np.ndarray, optional): Validation targets.

        Returns:
            float: The validation AUC score if validation data is provided, else training AUC.
        """
        # Capture feature names if input is a DataFrame
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()

        print("Training Random Forest model...")
        self.model.fit(X_train, y_train)

        # Evaluate on Training Data
        train_probs = self.model.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, train_probs)
        print(f"Training ROC AUC: {train_auc}")

        # Evaluate on Validation Data if provided
        val_auc = None
        if X_val is not None and y_val is not None:
            val_probs = self.model.predict_proba(X_val)[:, 1]
            val_auc = roc_auc_score(y_val, val_probs)
            print(f"Validation ROC AUC: {val_auc}")

        return val_auc if val_auc is not None else train_auc

    def predict_proba(self, X):
        """
        Generates probability predictions for the positive class (received pizza).

        Args:
            X (pd.DataFrame or np.ndarray): Features to predict on.

        Returns:
            np.ndarray: Array of probabilities for the positive class.
        """
        # predict_proba returns [prob_class_0, prob_class_1]
        # We return prob_class_1
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self):
        """
        Retrieves feature importance scores.

        Returns:
            pd.Series: Feature importances indexed by feature name (if available) or integer index.
        """
        importances = self.model.feature_importances_

        if self.feature_names:
            return pd.Series(importances, index=self.feature_names).sort_values(
                ascending=False
            )
        else:
            return pd.Series(importances).sort_values(ascending=False)
