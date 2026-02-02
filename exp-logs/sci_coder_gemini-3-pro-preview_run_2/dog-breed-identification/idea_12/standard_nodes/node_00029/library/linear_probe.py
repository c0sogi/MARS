import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import log_loss
from library.config import Config


class StreamClassifier:
    """
    A wrapper class for LogisticRegressionCV to act as a linear probe
    on top of pre-computed embeddings for a specific stream.

    This class handles the training (with internal cross-validation),
    persistence, and inference for the independent stream classifiers.
    """

    def __init__(self, stream_config):
        """
        Initialize the classifier for a specific stream.

        Args:
            stream_config (dict): Configuration dictionary for the stream (A or B).
                                  Used to determine naming for saved files.
        """
        self.stream_name = stream_config["name"]
        self.cache_prefix = stream_config["cache_prefix"]
        self.model_path = os.path.join(
            Config.WORKING_DIR, f"{self.cache_prefix}_logreg.joblib"
        )
        self.model = None

    def train(self, embeddings, labels):
        """
        Trains the LogisticRegressionCV model using the provided embeddings and labels.

        Args:
            embeddings (np.ndarray): Feature matrix of shape (n_samples, n_features).
            labels (np.ndarray): Target labels of shape (n_samples,).
        """
        print(f"Training Linear Probe for stream: {self.stream_name}")
        print(f"Input shape: {embeddings.shape}, Labels shape: {labels.shape}")

        # Load parameters from Config
        params = Config.LOGREG_PARAMS.copy()

        # Explicitly set scoring to neg_log_loss to ensure the internal CV
        # optimizes the competition metric.
        if "scoring" not in params:
            params["scoring"] = "neg_log_loss"

        # Initialize LogisticRegressionCV
        # This will automatically perform Cross-Validation to select the best C
        self.model = LogisticRegressionCV(**params)

        # Fit the model
        # Note: LogisticRegressionCV refits on the entire dataset using the best C found.
        self.model.fit(embeddings, labels)

        # Calculate training metrics on the refitted model
        preds_proba = self.model.predict_proba(embeddings)
        loss = log_loss(labels, preds_proba)

        print(f"Training finished for {self.stream_name}.")
        print(f"Selected C (inverse regularization strength): {self.model.C_}")
        print(f"Training Log Loss: {loss}")

        # Save the trained model
        self.save_model()

    def predict(self, embeddings):
        """
        Predicts class probabilities for the given embeddings.

        Args:
            embeddings (np.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Probability matrix of shape (n_samples, n_classes).
        """
        if self.model is None:
            self.load_model()

        return self.model.predict_proba(embeddings)

    def save_model(self):
        """Saves the internal sklearn model to disk."""
        if self.model is None:
            print("No model to save.")
            return

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        joblib.dump(self.model, self.model_path)
        print(f"Model saved to {self.model_path}")

    def load_model(self):
        """Loads the sklearn model from disk."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. Train the model first."
            )

        self.model = joblib.load(self.model_path)
        print(f"Model loaded from {self.model_path}")

    @property
    def classes_(self):
        """
        Expose classes_ attribute from the underlying model.
        Useful for mapping probability columns to breed names.
        """
        if self.model is None:
            self.load_model()
        return self.model.classes_
