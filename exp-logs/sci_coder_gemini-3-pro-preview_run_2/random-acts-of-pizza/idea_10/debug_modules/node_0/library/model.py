import os
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import setup_logger, set_seed
from library.features import generate_features

# Initialize Logger
logger = setup_logger("model")


class BaggedLREnsemble:
    """
    A Bagged Ensemble of Logistic Regression models.
    Performs internal cross-validation to optimize the regularization parameter C
    before fitting the final ensemble.
    """

    def __init__(self):
        self.model = None
        self.best_c = None

    def optimize_and_fit(self, X, y):
        """
        Optimizes the Logistic Regression regularization parameter C using CV,
        then fits a BaggingClassifier with the optimal base estimator.

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training labels.
        """
        logger.info("Starting hyperparameter optimization for Logistic Regression C...")

        # 1. Hyperparameter Search (Grid Search C)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=Config.SEED)
        best_score = -np.inf
        best_c = Config.LOGREG_C_GRID[0]

        for c in Config.LOGREG_C_GRID:
            # Base estimator for CV
            clf = LogisticRegression(
                C=c,
                class_weight="balanced",
                solver="lbfgs",
                max_iter=1000,
                random_state=Config.SEED,
            )

            # Cross-validation
            # n_jobs=-1 uses all available processors
            scores = cross_val_score(clf, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
            mean_score = np.mean(scores)
            std_score = np.std(scores)

            logger.info(f"C={c}: Mean AUC = {mean_score:.10f} (+/- {std_score:.10f})")

            if mean_score > best_score:
                best_score = mean_score
                best_c = c

        self.best_c = best_c
        logger.info(
            f"Optimization complete. Best C: {self.best_c} with AUC: {best_score:.10f}"
        )

        # 2. Fit Final Ensemble
        logger.info("Fitting final Bagged Ensemble with optimal C...")

        # Define the optimized base estimator
        base_estimator = LogisticRegression(
            C=self.best_c,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=1000,
            random_state=Config.SEED,
        )

        # Initialize Bagging Classifier
        # Note: 'estimator' parameter is used in scikit-learn >= 1.2
        self.model = BaggingClassifier(
            estimator=base_estimator, **Config.BAGGING_PARAMS
        )

        self.model.fit(X, y)
        logger.info("Model fitting complete.")

    def predict_proba(self, X):
        """
        Predicts class probabilities for the input data.

        Args:
            X (np.ndarray): Input features.

        Returns:
            np.ndarray: Probabilities for the positive class (class 1).
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")

        # predict_proba returns [n_samples, n_classes], we want the probability of class 1
        return self.model.predict_proba(X)[:, 1]

    def save(self, filepath):
        """Saves the trained model to disk."""
        joblib.dump(self, filepath)
        logger.info(f"Model saved to {filepath}")

    @staticmethod
    def load(filepath):
        """Loads a trained model from disk."""
        return joblib.load(filepath)


def run_training_pipeline():
    """
    Orchestrates the full training pipeline:
    1. Feature Generation
    2. Model Optimization and Training
    3. Validation Evaluation
    4. Submission Generation
    """
    set_seed(Config.SEED)

    # 1. Generate/Load Features
    logger.info("Step 1: Retrieving Features...")
    X_train, y_train, X_val, y_val, X_test, test_ids = generate_features(
        load_cached_data=True
    )

    # 2. Initialize and Train Model
    logger.info("Step 2: Training Model...")
    ensemble = BaggedLREnsemble()
    ensemble.optimize_and_fit(X_train, y_train)

    # 3. Validation Evaluation
    logger.info("Step 3: Evaluating on Validation Set...")
    val_probs = ensemble.predict_proba(X_val)
    val_auc = roc_auc_score(y_val, val_probs)

    print(f"Validation ROC AUC: {val_auc:.15f}")
    logger.info(f"Validation ROC AUC: {val_auc:.15f}")

    # 4. Generate Submission
    logger.info("Step 4: Generating Submission...")
    test_probs = ensemble.predict_proba(X_test)

    # Create submission DataFrame
    submission_df = pd.DataFrame(
        {Config.ID_COL: test_ids, Config.TARGET_COL: test_probs}
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # Save submission
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")

    # Save model for future use
    model_path = os.path.join(Config.WORKING_DIR, "bagged_lr_ensemble.joblib")
    ensemble.save(model_path)
