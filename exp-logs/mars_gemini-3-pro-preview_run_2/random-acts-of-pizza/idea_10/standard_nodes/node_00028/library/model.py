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
    A Cross-Validation Ensemble of Bagged Logistic Regression models.
    Implements 'CV-Bagging': trains K Bagged models on K folds and averages predictions.
    This reduces variance and improves generalization (Cite Lesson 27).
    """

    def __init__(self):
        self.models = []
        self.best_c = None

    def optimize_and_fit(self, X, y):
        """
        Optimizes C using CV, then trains a Bagged Ensemble on EACH fold of the data.
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

        # 2. Fit CV Ensemble (Train K models on K folds)
        logger.info("Fitting CV-Bagging Ensemble (5 models)...")
        self.models = []

        # Re-use SKF to ensure consistent splits
        for fold_idx, (train_idx, _) in enumerate(skf.split(X, y)):
            X_fold, y_fold = X[train_idx], y[train_idx]

            base_estimator = LogisticRegression(
                C=self.best_c,
                class_weight="balanced",
                solver="lbfgs",
                max_iter=1000,
                random_state=Config.SEED + fold_idx,  # Vary seed slightly per fold
            )

            model = BaggingClassifier(estimator=base_estimator, **Config.BAGGING_PARAMS)

            model.fit(X_fold, y_fold)
            self.models.append(model)

        logger.info(f"Fitted {len(self.models)} models.")

    def predict_proba(self, X):
        """
        Predicts class probabilities by averaging predictions from all CV models.
        """
        if not self.models:
            raise RuntimeError("Models have not been fitted yet.")

        # Collect predictions from all models
        all_probs = []
        for model in self.models:
            probs = model.predict_proba(X)[:, 1]
            all_probs.append(probs)

        # Average predictions
        avg_probs = np.mean(all_probs, axis=0)
        return avg_probs

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
