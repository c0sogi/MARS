import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from library.config import Config
from library.data_processing import TextPreprocessor
from library.utils import clip_probabilities


class TfidfExpert:
    """
    The Stylometric Expert model using TF-IDF features and Logistic Regression.
    This class handles feature extraction (delegated to TextPreprocessor) and
    model training/inference.
    """

    def __init__(self):
        # Initialize the Logistic Regression model with parameters from Config
        self.model = LogisticRegression(**Config.LOGREG_PARAMS)
        # Initialize the preprocessor for feature extraction
        self.preprocessor = TextPreprocessor()

    def get_features(self, train_text, val_text, test_text, load_cached_data=True):
        """
        Generates or loads TF-IDF features for train, validation, and test sets.
        Delegates to library.data_processing.TextPreprocessor which handles caching.

        Args:
            train_text (list or pd.Series): Training text samples.
            val_text (list or pd.Series): Validation text samples.
            test_text (list or pd.Series): Test text samples.
            load_cached_data (bool): Whether to load from cache if available.

        Returns:
            tuple: (X_train, X_val, X_test) sparse matrices.
        """
        return self.preprocessor.get_tfidf_features(
            train_text, val_text, test_text, load_cached_data=load_cached_data
        )

    def fit(self, X, y):
        """
        Fits the Logistic Regression model on the provided features and labels.

        Args:
            X (sparse matrix): Feature matrix.
            y (array-like): Target labels (integers corresponding to Config.LABEL2ID).
        """
        print("Fitting Logistic Regression model...")
        self.model.fit(X, y)

    def predict_proba(self, X):
        """
        Predicts class probabilities for the given features.

        Args:
            X (sparse matrix): Feature matrix.

        Returns:
            np.ndarray: Predicted probabilities of shape (n_samples, n_classes).
        """
        return self.model.predict_proba(X)

    def validate(self, X_val, y_val):
        """
        Evaluates the model on the validation set and prints metrics.

        Args:
            X_val (sparse matrix): Validation features.
            y_val (array-like): Validation labels (integers).

        Returns:
            tuple: (log_loss, accuracy)
        """
        print("Validating model...")
        probs = self.predict_proba(X_val)

        # Clip probabilities to avoid log(0) and match metric definition
        clipped_probs = clip_probabilities(probs)

        # Calculate Log Loss
        # We explicitly provide labels to ensure correct calculation even if a class is missing in y_val
        labels = list(range(len(Config.LABELS)))
        loss = log_loss(y_val, clipped_probs, labels=labels)

        # Calculate Accuracy
        preds = np.argmax(probs, axis=1)
        acc = accuracy_score(y_val, preds)

        # Print full precision metrics
        print(f"Validation Log Loss: {loss}")
        print(f"Validation Accuracy: {acc}")

        return loss, acc
