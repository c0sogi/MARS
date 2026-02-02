import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from library.config import C_GRID, SEED, SUBMISSION_FILE, TEST_META_PATH
from library.model_dispatcher import get_logistic_regression
from library.utils import compute_auc, set_seed


class Trainer:
    """
    Manages the training pipeline including hyperparameter tuning (CV),
    final model training, and submission generation.
    """

    def __init__(self):
        set_seed(SEED)
        self.best_C = None
        self.best_auc = -1.0
        self.model = None

    def run_cross_validation(
        self, X_train: np.ndarray, y_train: np.ndarray, k_folds: int = 5
    ) -> float:
        """
        Performs Stratified K-Fold Cross-Validation to select the best regularization parameter C.

        Args:
            X_train (np.ndarray): Training feature matrix.
            y_train (np.ndarray): Training labels.
            k_folds (int): Number of cross-validation folds.

        Returns:
            float: The best C value found.
        """
        print(f"Starting Cross-Validation with {k_folds} folds...")
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=SEED)

        results = {}

        for C in C_GRID:
            fold_aucs = []

            # Iterate through folds
            for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
                X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

                # Initialize and train model
                model = get_logistic_regression(C=C)
                model.fit(X_fold_train, y_fold_train)

                # Predict probabilities for the positive class (1)
                preds = model.predict_proba(X_fold_val)[:, 1]

                # Evaluate
                auc = compute_auc(y_fold_val, preds)
                fold_aucs.append(auc)

            mean_auc = np.mean(fold_aucs)
            std_auc = np.std(fold_aucs)
            results[C] = mean_auc

            print(f"C: {C}, Mean AUC: {mean_auc}, Std AUC: {std_auc}")

        # Select best C based on highest Mean AUC
        self.best_C = max(results, key=results.get)
        self.best_auc = results[self.best_C]

        print(f"Best C selected: {self.best_C} with Mean CV AUC: {self.best_auc}")
        return self.best_C

    def train_final_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ):
        """
        Retrains the final model on the combined training and validation dataset
        using the optimal C parameter found during cross-validation.

        Args:
            X_train (np.ndarray): Training feature matrix.
            y_train (np.ndarray): Training labels.
            X_val (np.ndarray): Validation feature matrix.
            y_val (np.ndarray): Validation labels.
        """
        if self.best_C is None:
            raise ValueError("Run cross-validation first to select hyperparameter C.")

        print("Retraining final model on full dataset (Train + Val)...")

        # Combine datasets to maximize training data
        X_full = np.vstack([X_train, X_val])
        y_full = np.concatenate([y_train, y_val])

        # Train final model
        self.model = get_logistic_regression(C=self.best_C)
        self.model.fit(X_full, y_full)

        print("Final model training complete.")

    def generate_submission(self, X_test: np.ndarray):
        """
        Generates predictions for the test set and saves the submission file.

        Args:
            X_test (np.ndarray): Test feature matrix.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        print("Generating predictions for test set...")

        # Predict probabilities for class 1 (received pizza)
        test_preds = self.model.predict_proba(X_test)[:, 1]

        # Load test metadata to get request_ids
        # We rely on the fact that data_loader processes files sequentially based on the metadata CSV
        if not os.path.exists(TEST_META_PATH):
            raise FileNotFoundError(f"Test metadata file not found at {TEST_META_PATH}")

        df_test_meta = pd.read_csv(TEST_META_PATH)

        if len(df_test_meta) != len(test_preds):
            raise ValueError(
                f"Mismatch in test set size: Metadata {len(df_test_meta)} vs Predictions {len(test_preds)}"
            )

        # Create submission DataFrame
        submission = pd.DataFrame(
            {
                "request_id": df_test_meta["request_id"],
                "requester_received_pizza": test_preds,
            }
        )

        # Save to disk
        os.makedirs(os.path.dirname(SUBMISSION_FILE), exist_ok=True)
        submission.to_csv(SUBMISSION_FILE, index=False)
        print(f"Submission saved to {SUBMISSION_FILE}")
