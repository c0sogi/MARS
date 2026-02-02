import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from library.config import Config


class BirdRandomForest:
    """
    Wrapper for a Multi-Output Random Forest Classifier.
    Handles initialization, training, prediction formatting, and persistence.
    """

    def __init__(self):
        """
        Initialize the Random Forest model with hyperparameters from Config.
        """
        self.model = RandomForestClassifier(**Config.RF_PARAMS)
        self.num_species = Config.NUM_SPECIES

    def fit(self, X, y):
        """
        Trains the Random Forest model on the provided data.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Multi-label target matrix of shape (n_samples, n_species).
        """
        self.model.fit(X, y)

    def predict_proba(self, X):
        """
        Predicts the probability of presence for each species.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Probability matrix of shape (n_samples, n_species).
        """
        # sklearn's predict_proba for multi-label data returns a list of arrays,
        # one for each target (species). Each array is (n_samples, n_classes_for_target).
        probas_list = self.model.predict_proba(X)

        # We need to extract the probability of the positive class (index 1) for each species.
        final_probas = []

        # Handle edge case where num_species=1 (returns array instead of list)
        if not isinstance(probas_list, list) and self.num_species == 1:
            probas_list = [probas_list]

        for i, class_proba in enumerate(probas_list):
            # class_proba is typically (n_samples, 2) for binary targets [Prob(0), Prob(1)]

            if class_proba.shape[1] == 2:
                # Standard case: both 0 and 1 labels existed in training
                final_probas.append(class_proba[:, 1])

            elif class_proba.shape[1] == 1:
                # Edge case: Only one label (either all 0s or all 1s) was present in training.
                # We check self.model.classes_[i] to see which label was present.
                classes_seen = self.model.classes_[i]

                if classes_seen[0] == 1:
                    # Only 1s were seen -> Probability is 1.0
                    final_probas.append(np.ones(X.shape[0]))
                else:
                    # Only 0s were seen -> Probability is 0.0
                    final_probas.append(np.zeros(X.shape[0]))
            else:
                # Fallback for unexpected shapes (should not occur in binary classification)
                final_probas.append(class_proba[:, -1])

        # Stack arrays column-wise to get (n_samples, n_species)
        return np.column_stack(final_probas)

    def save(self, filepath):
        """
        Saves the trained model to the specified filepath.

        Args:
            filepath (str): Path to save the model (e.g., .pkl or .joblib).
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)

    @classmethod
    def load(cls, filepath):
        """
        Loads a trained model from the specified filepath.

        Args:
            filepath (str): Path to the saved model.

        Returns:
            BirdRandomForest: Instance with the loaded model.
        """
        instance = cls()
        if os.path.exists(filepath):
            instance.model = joblib.load(filepath)
        else:
            raise FileNotFoundError(f"Model file not found at {filepath}")
        return instance
