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
        self.c_values = Config.LR_CS
        self.base_params = {
            "solver": Config.LR_SOLVER,
            "max_iter": Config.LR_MAX_ITER,
            "n_jobs": Config.N_JOBS,
            "random_state": Config.SEED,
            "verbose": 0,
        }

    def train(self, X_train, y_train, X_val, y_val):
        """
        Trains the One-Vs-Rest Logistic Regression models (Ensemble) and evaluates on validation data.

        Args:
            X_train (scipy.sparse.csr_matrix): Training features.
            y_train (pd.DataFrame): Training labels.
            X_val (scipy.sparse.csr_matrix): Validation features.
            y_val (pd.DataFrame): Validation labels.

        Returns:
            float: The mean column-wise ROC AUC score.
        """
        self.logger.info("Starting training of Logistic Regression Ensemble...")
        self.logger.info(f"C Values: {self.c_values}")

        auc_scores = []

        for label in self.target_cols:
            self.logger.info(f"Training ensemble for label: {label}")

            label_models = []
            val_preds_ensemble = np.zeros(X_val.shape[0])

            y_train_target = y_train[label].values
            y_val_target = y_val[label].values

            for c in self.c_values:
                # Initialize model with specific C
                params = self.base_params.copy()
                params["C"] = c
                model = LogisticRegression(**params)

                # Fit model
                model.fit(X_train, y_train_target)

                # Accumulate predictions
                val_preds_c = model.predict_proba(X_val)[:, 1]
                val_preds_ensemble += val_preds_c

                label_models.append(model)

            # Average predictions
            val_preds_ensemble /= len(self.c_values)

            # Calculate ROC AUC
            score = roc_auc_score(y_val_target, val_preds_ensemble)
            auc_scores.append(score)

            # Log score with full precision
            self.logger.info(f"Label: {label} - Ensemble Validation AUC: {score}")

            # Store trained models
            self.models[label] = label_models

        mean_auc = np.mean(auc_scores)
        self.logger.info(f"Training Complete. Mean Column-wise ROC AUC: {mean_auc}")

        return mean_auc

    def predict_proba(self, X):
        """
        Predicts probabilities for all toxicity labels using the ensemble.

        Args:
            X (scipy.sparse.csr_matrix): Features to predict on.

        Returns:
            pd.DataFrame: DataFrame containing probabilities for each label.
        """
        if not self.models:
            raise RuntimeError("Models have not been trained. Call train() first.")

        predictions = {}
        for label in self.target_cols:
            label_models = self.models[label]
            ensemble_preds = np.zeros(X.shape[0])

            for model in label_models:
                ensemble_preds += model.predict_proba(X)[:, 1]

            predictions[label] = ensemble_preds / len(label_models)

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
