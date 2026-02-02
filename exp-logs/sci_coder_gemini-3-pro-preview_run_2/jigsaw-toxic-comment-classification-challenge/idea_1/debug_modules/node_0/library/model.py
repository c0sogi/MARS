import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.utils import setup_logger


class ToxicityClassifier:
    """
    A Multi-Label Classifier using One-Vs-Rest Logistic Regression.
    Trains a separate classifier for each toxicity label.
    """

    def __init__(self):
        """
        Initializes the classifier with configuration parameters.
        """
        self.logger = setup_logger("model")
        self.models = {}
        self.target_cols = Config.TARGET_COLS
        self.params = {
            "C": Config.LR_C,
            "solver": Config.LR_SOLVER,
            "max_iter": Config.LR_MAX_ITER,
            "n_jobs": Config.N_JOBS,
            "random_state": Config.SEED,
            "verbose": 0,
        }

    def train(self, X_train, y_train, X_val, y_val):
        """
        Trains the One-Vs-Rest Logistic Regression models and evaluates on validation data.

        Args:
            X_train (scipy.sparse.csr_matrix): Training features.
            y_train (pd.DataFrame): Training labels.
            X_val (scipy.sparse.csr_matrix): Validation features.
            y_val (pd.DataFrame): Validation labels.

        Returns:
            float: The mean column-wise ROC AUC score.
        """
        self.logger.info("Starting training of Logistic Regression models...")
        self.logger.info(f"Hyperparameters: {self.params}")

        auc_scores = []

        for label in self.target_cols:
            self.logger.info(f"Training model for label: {label}")

            # Initialize model
            model = LogisticRegression(**self.params)

            # Extract specific label vectors
            # Ensure we are working with 1D arrays
            y_train_target = y_train[label].values
            y_val_target = y_val[label].values

            # Fit model
            model.fit(X_train, y_train_target)

            # Predict probabilities on validation set (class 1)
            val_preds = model.predict_proba(X_val)[:, 1]

            # Calculate ROC AUC
            score = roc_auc_score(y_val_target, val_preds)
            auc_scores.append(score)

            # Log score with full precision
            self.logger.info(f"Label: {label} - Validation AUC: {score}")

            # Store trained model
            self.models[label] = model

        mean_auc = np.mean(auc_scores)
        self.logger.info(f"Training Complete. Mean Column-wise ROC AUC: {mean_auc}")

        return mean_auc

    def predict_proba(self, X):
        """
        Predicts probabilities for all toxicity labels.

        Args:
            X (scipy.sparse.csr_matrix): Features to predict on.

        Returns:
            pd.DataFrame: DataFrame containing probabilities for each label.
        """
        if not self.models:
            raise RuntimeError("Models have not been trained. Call train() first.")

        predictions = {}
        for label in self.target_cols:
            model = self.models[label]
            # Store probability of the positive class
            predictions[label] = model.predict_proba(X)[:, 1]

        return pd.DataFrame(predictions)

    def generate_submission(self, X_test, test_ids):
        """
        Generates predictions for the test set and saves them to the submission file.

        Args:
            X_test (scipy.sparse.csr_matrix): Test features.
            test_ids (pd.Series or list): IDs corresponding to the test features.
        """
        self.logger.info("Generating predictions for test set...")

        # Generate probabilities
        probs_df = self.predict_proba(X_test)

        # Construct submission DataFrame
        submission_df = pd.DataFrame()

        # Ensure IDs are assigned correctly (using values to ignore index alignment)
        submission_df["id"] = (
            test_ids.values if hasattr(test_ids, "values") else test_ids
        )

        # Assign probability columns in the correct order
        for label in self.target_cols:
            submission_df[label] = probs_df[label]

        # Save to disk
        save_path = Config.SUBMISSION_PATH
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        submission_df.to_csv(save_path, index=False)
        self.logger.info(f"Submission saved to {save_path}")
        self.logger.info(f"Submission shape: {submission_df.shape}")
