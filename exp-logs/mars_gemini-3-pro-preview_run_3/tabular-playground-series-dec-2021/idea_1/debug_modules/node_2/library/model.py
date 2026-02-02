import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from library.config import Config


class GBDTWrapper:
    """
    A wrapper class for the Gradient Boosting Decision Tree (XGBoost) model.
    Encapsulates initialization, training with early stopping, and submission generation.
    """

    def __init__(self, **kwargs):
        """
        Initialize the XGBoost Classifier.

        Args:
            **kwargs: Arbitrary keyword arguments to override default Config.MODEL_PARAMS.
                      Useful for adjusting n_estimators or learning_rate dynamically.
        """
        # Load default parameters from Config
        self.params = Config.MODEL_PARAMS.copy()

        # Update with any overrides
        self.params.update(kwargs)

        # Ensure output directories exist
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

        # Initialize the model
        self.model = xgb.XGBClassifier(**self.params)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains the model using the provided training data.
        Implements early stopping if validation data is provided.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (np.ndarray): Training targets (0-indexed).
            X_val (pd.DataFrame, optional): Validation features.
            y_val (np.ndarray, optional): Validation targets (0-indexed).
        """
        eval_set = []

        # Add training set to evaluation to monitor training error
        if X_train is not None and y_train is not None:
            eval_set.append((X_train, y_train))

        # Add validation set for early stopping
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        print(f"Starting training with params: {self.params}")

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            early_stopping_rounds=Config.EARLY_STOPPING_ROUNDS,
            verbose=Config.VERBOSE_EVAL,
        )

        # Calculate and print full precision validation metric
        if X_val is not None and y_val is not None:
            val_preds = self.model.predict(X_val)
            acc = accuracy_score(y_val, val_preds)
            print(f"Final Validation Accuracy: {acc}")

        # Save the trained model
        self.model.save_model(Config.MODEL_OUTPUT_PATH)
        print(f"Model saved to {Config.MODEL_OUTPUT_PATH}")

    def predict(self, X):
        """
        Generates predictions for the given features.

        Args:
            X (pd.DataFrame): Features to predict on.

        Returns:
            np.ndarray: Predicted class labels (0-indexed).
        """
        return self.model.predict(X)

    def generate_submission(self, X_test, test_ids):
        """
        Generates predictions for the test set and saves the submission file.
        Handles mapping from 0-indexed classes back to 1-indexed targets.

        Args:
            X_test (pd.DataFrame): Test features.
            test_ids (pd.Series or list): IDs corresponding to the test features.
        """
        print("Generating predictions for submission...")

        # Generate raw predictions (0-6)
        preds = self.predict(X_test)

        # Map predictions back to original class range (1-7)
        # The data loader subtracted 1, so we add 1 back.
        preds_mapped = preds + 1

        # Create submission DataFrame
        submission = pd.DataFrame(
            {Config.ID_COL: test_ids, Config.TARGET_COL: preds_mapped.astype(int)}
        )

        # Save to CSV
        submission.to_csv(Config.SUBMISSION_FILE_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE_PATH}")
