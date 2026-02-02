import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from library.config import Config
from library.utils import compute_metric


class LinearEnsembleTrainer:
    """
    Trains an ensemble of Logistic Regression models with varying regularization strengths (C).
    This constitutes 'Branch A' of the hybrid solution.
    """

    def __init__(self):
        self.c_values = Config.LR_C_VALUES
        self.solver = Config.LR_SOLVER
        self.max_iter = Config.LR_MAX_ITER
        self.n_jobs = Config.LR_N_JOBS
        self.labels = Config.LABEL_COLS
        self.seed = Config.SEED

    def train_and_predict(self, X_train, y_train, X_val, y_val, X_test):
        """
        Trains the ensemble and generates predictions for validation and test sets.

        Args:
            X_train: Sparse matrix of training features.
            y_train: DataFrame of training labels.
            X_val: Sparse matrix of validation features.
            y_val: DataFrame of validation labels.
            X_test: Sparse matrix of test features.

        Returns:
            tuple: (val_preds, test_preds) - Numpy arrays of shape (n_samples, n_labels)
                   containing the averaged probabilities from the ensemble.
        """
        print("Starting Linear Ensemble Training...")

        # Initialize arrays to store aggregated predictions
        # We use zeros and add probabilities, then divide by num_models later
        val_preds_global = np.zeros((X_val.shape[0], len(self.labels)))
        test_preds_global = np.zeros((X_test.shape[0], len(self.labels)))

        # Iterate over each target label (One-Vs-Rest strategy)
        for i, label in enumerate(self.labels):
            print(f"Processing label: {label}")

            # Extract target vectors
            y_train_col = y_train[label]
            y_val_col = y_val[label]

            # Accumulators for the current label across different C values
            val_preds_label_sum = np.zeros(X_val.shape[0])
            test_preds_label_sum = np.zeros(X_test.shape[0])

            # Train a model for each C value in the ensemble configuration
            for c in self.c_values:
                # Initialize model
                # We use class_weight=None as the metric is AUC, which is insensitive to imbalance,
                # and we want calibrated probabilities.
                model = LogisticRegression(
                    C=c,
                    solver=self.solver,
                    max_iter=self.max_iter,
                    n_jobs=self.n_jobs,
                    random_state=self.seed,
                )

                # Fit model
                model.fit(X_train, y_train_col)

                # Predict probabilities (index 1 is the positive class)
                # We accumulate them directly
                val_preds_label_sum += model.predict_proba(X_val)[:, 1]
                test_preds_label_sum += model.predict_proba(X_test)[:, 1]

            # Average the predictions
            num_models = len(self.c_values)
            val_preds_label_avg = val_preds_label_sum / num_models
            test_preds_label_avg = test_preds_label_sum / num_models

            # Store in global arrays
            val_preds_global[:, i] = val_preds_label_avg
            test_preds_global[:, i] = test_preds_label_avg

            # Calculate and print metric for this label
            # We use the provided utility, passing the specific column
            try:
                score = compute_metric(y_val_col, val_preds_label_avg)
                print(f"Label {label} Ensemble Val AUC: {score}")
            except Exception as e:
                print(f"Could not compute metric for {label}: {e}")

        # Calculate overall metric
        print("Computing overall Linear Ensemble validation metric...")
        overall_score = compute_metric(y_val[self.labels], val_preds_global)
        print(f"Overall Linear Ensemble Val AUC: {overall_score}")

        return val_preds_global, test_preds_global
