import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from library.config import Config


class RFModelWrapper:
    """
    Wrapper for the Stream A: Scope-Restricted Sentiment-Aware Random Forest.

    This model explicitly excludes high-cardinality/sparse subreddit history features
    to avoid noise, focusing instead on:
    1. High-Fidelity TF-IDF (Lexical signals)
    2. Full-Spectrum Metadata (Numerical magnitudes, ratios)
    3. Sentiment & Subjectivity Scores (Dense text signals)
    """

    def __init__(self):
        # Initialize Random Forest with configuration parameters
        # (n_estimators=500, min_samples_leaf=1, class_weight='balanced', etc.)
        self.model = RandomForestClassifier(**Config.RF_PARAMS)

    def _prepare_X(self, tfidf_data, meta_data):
        """
        Combines sparse TF-IDF features with dense metadata features.

        Args:
            tfidf_data (scipy.sparse.csr_matrix): Sparse TF-IDF matrix.
            meta_data (numpy.ndarray): Dense metadata array (scaled/imputed).

        Returns:
            scipy.sparse.csr_matrix: Combined feature matrix.
        """
        # Stack sparse and dense features horizontally
        # This creates the final input vector for the Tree model
        return hstack([tfidf_data, meta_data]).tocsr()

    def train(self, rf_data, labels):
        """
        Trains the Random Forest model and evaluates on validation set.

        Args:
            rf_data (dict): Dictionary containing 'train_tfidf', 'train_meta',
                            'val_tfidf', 'val_meta'.
            labels (dict): Dictionary containing 'y_train', 'y_val'.

        Returns:
            float: Validation ROC AUC score.
        """
        print("Constructing RF Training Set...")
        X_train = self._prepare_X(rf_data["train_tfidf"], rf_data["train_meta"])
        y_train = labels["y_train"]

        print(
            f"Training Random Forest with {X_train.shape[0]} samples and {X_train.shape[1]} features..."
        )
        self.model.fit(X_train, y_train)

        # Validation
        val_score = 0.0
        if "val_tfidf" in rf_data and "val_meta" in rf_data and "y_val" in labels:
            print("Validating Random Forest...")
            X_val = self._prepare_X(rf_data["val_tfidf"], rf_data["val_meta"])
            y_val = labels["y_val"]

            # Predict probabilities for the positive class
            val_preds = self.model.predict_proba(X_val)[:, 1]

            val_score = roc_auc_score(y_val, val_preds)
            print(f"RF Validation ROC AUC: {val_score}")

        return val_score

    def predict(self, rf_data):
        """
        Generates predictions for the test set.

        Args:
            rf_data (dict): Dictionary containing 'test_tfidf', 'test_meta'.

        Returns:
            numpy.ndarray: Predicted probabilities for the positive class.
        """
        print("Constructing RF Test Set...")
        X_test = self._prepare_X(rf_data["test_tfidf"], rf_data["test_meta"])

        print("Generating RF Predictions...")
        test_preds = self.model.predict_proba(X_test)[:, 1]

        return test_preds
