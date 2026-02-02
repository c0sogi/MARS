import os
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score

from library import config, utils

# Setup logger
logger = utils.setup_logger("classifier")


class PizzaClassifier:
    """
    Implements the linear classification stage using Logistic Regression.
    Includes hyperparameter optimization via Grid Search and submission generation.
    """

    def __init__(self):
        self.model = None
        self.best_params = None
        # Path to save the trained model artifact
        self.model_path = os.path.join(config.WORKING_DIR, "logistic_regression.joblib")

    def optimize(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Optimizes the Logistic Regression hyperparameters using GridSearchCV
        and trains the final model on the provided training data.

        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training labels.
        """
        logger.info(
            f"Starting optimization with {X_train.shape[0]} samples and {X_train.shape[1]} features..."
        )

        # Define the base model
        # We use class_weight='balanced' to handle the 1:3 imbalance
        clf = LogisticRegression(
            solver=config.CLASSIFIER_SOLVER,
            max_iter=config.CLASSIFIER_MAX_ITER,
            class_weight="balanced",
            random_state=config.SEED,
        )

        # Define the parameter grid from config
        param_grid = {"C": config.CLASSIFIER_C_GRID}

        # Define Cross-Validation strategy
        cv = StratifiedKFold(
            n_splits=config.CV_FOLDS, shuffle=True, random_state=config.SEED
        )

        # Initialize GridSearchCV
        grid_search = GridSearchCV(
            estimator=clf,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            verbose=0,  # Silent execution
        )

        # Execute Grid Search
        grid_search.fit(X_train, y_train)

        # Store results
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        logger.info("Optimization complete.")
        logger.info(f"Best Parameters: {self.best_params}")
        # Print full precision as requested
        print(f"Best CV ROC AUC: {best_score}")

        # Save the model
        self.save_model()

    def save_model(self):
        """Saves the current model to disk."""
        if self.model is not None:
            joblib.dump(self.model, self.model_path)
            logger.info(f"Model saved to {self.model_path}")
        else:
            logger.warning("No model to save.")

    def load_model(self):
        """Loads the model from disk if it exists."""
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            logger.info(f"Model loaded from {self.model_path}")
        else:
            logger.warning(f"Model file not found at {self.model_path}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Generates probability predictions for the positive class.

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Probabilities for class 1.
        """
        if self.model is None:
            self.load_model()
            if self.model is None:
                raise RuntimeError("Model is not trained or loaded.")

        # predict_proba returns shape (n_samples, 2), we want column 1
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X_val: np.ndarray, y_val: np.ndarray):
        """
        Evaluates the model on the validation set.

        Args:
            X_val (np.ndarray): Validation features.
            y_val (np.ndarray): Validation labels.
        """
        logger.info("Evaluating on validation set...")
        probs = self.predict_proba(X_val)
        auc = roc_auc_score(y_val, probs)

        # Print full precision
        print(f"Validation ROC AUC: {auc}")
        return auc

    def generate_submission(self, X_test: np.ndarray):
        """
        Generates the submission file for the test set.

        Args:
            X_test (np.ndarray): Test features.
        """
        logger.info("Generating submission file...")

        # Generate predictions
        probs = self.predict_proba(X_test)

        # Load Test Metadata to retrieve request_ids
        # The order of X_test matches the order in metadata/test.csv
        if not os.path.exists(config.TEST_META_PATH):
            raise FileNotFoundError(
                f"Test metadata file missing: {config.TEST_META_PATH}"
            )

        df_test_meta = pd.read_csv(config.TEST_META_PATH)

        if len(df_test_meta) != len(probs):
            logger.error(
                f"Mismatch: Metadata has {len(df_test_meta)} rows, Predictions have {len(probs)}."
            )
            raise ValueError("Test metadata and prediction lengths do not match.")

        # Create submission DataFrame
        submission_df = pd.DataFrame(
            {config.ID_COL: df_test_meta[config.ID_COL], config.TARGET_COL: probs}
        )

        # Ensure submission directory exists
        os.makedirs(os.path.dirname(config.SUBMISSION_PATH), exist_ok=True)

        # Save to CSV
        submission_df.to_csv(config.SUBMISSION_PATH, index=False)
        logger.info(f"Submission saved to {config.SUBMISSION_PATH}")
