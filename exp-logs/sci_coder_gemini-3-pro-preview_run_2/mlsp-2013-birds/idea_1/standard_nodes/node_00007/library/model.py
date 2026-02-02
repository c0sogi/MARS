import os
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.dummy import DummyClassifier
from library.config import Config


class BirdModel:
    """
    Wrapper for a set of Independent XGBoost Classifiers (Binary Relevance).
    Handles initialization, training with dynamic class weighting, prediction, and persistence.
    """

    def __init__(self):
        """
        Initialize the container for species-specific models.
        """
        self.models = []
        self.num_species = Config.NUM_SPECIES

    def fit(self, X, y):
        """
        Trains independent models for each species with dynamic class balancing.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Multi-label target matrix of shape (n_samples, n_species).
        """
        self.models = []
        for i in range(self.num_species):
            y_i = y[:, i]

            # Calculate class distribution
            num_pos = np.sum(y_i)
            num_neg = len(y_i) - num_pos

            # Handle edge cases where a species is absent or omnipresent in training split
            if num_pos == 0 or num_neg == 0:
                # Use a dummy classifier that predicts the constant class
                clf = DummyClassifier(strategy="constant", constant=int(num_pos > 0))
                clf.fit(X, y_i)
            else:
                # Calculate scale_pos_weight for XGBoost to handle imbalance
                scale_pos_weight = num_neg / num_pos

                params = Config.XGB_PARAMS.copy()
                params["scale_pos_weight"] = scale_pos_weight

                clf = XGBClassifier(**params)
                clf.fit(X, y_i)

            self.models.append(clf)

    def predict_proba(self, X):
        """
        Predicts the probability of presence for each species.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Probability matrix of shape (n_samples, n_species).
        """
        probas = []
        for clf in self.models:
            p = clf.predict_proba(X)

            # Handle output shape differences between XGB and DummyClassifier
            if p.shape[1] == 2:
                # Standard binary case: [Prob(0), Prob(1)]
                probas.append(p[:, 1])
            else:
                # Single class case (DummyClassifier or degenerated XGB)
                # clf.classes_ tells us which class is present
                if clf.classes_[0] == 1:
                    probas.append(np.ones(X.shape[0]))
                else:
                    probas.append(np.zeros(X.shape[0]))

        # Stack arrays column-wise to get (n_samples, n_species)
        return np.column_stack(probas)

    def save(self, filepath):
        """
        Saves the list of trained models to the specified filepath.

        Args:
            filepath (str): Path to save the model list.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.models, filepath)

    @classmethod
    def load(cls, filepath):
        """
        Loads the trained models from the specified filepath.

        Args:
            filepath (str): Path to the saved model.

        Returns:
            BirdModel: Instance with the loaded models.
        """
        instance = cls()
        if os.path.exists(filepath):
            instance.models = joblib.load(filepath)
        else:
            raise FileNotFoundError(f"Model file not found at {filepath}")
        return instance
