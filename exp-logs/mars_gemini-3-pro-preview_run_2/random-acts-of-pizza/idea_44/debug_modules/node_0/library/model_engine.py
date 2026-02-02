import os
import joblib
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from library.config import Config
from library.utils import setup_logger

# Initialize Logger
logger = setup_logger(
    "model_engine", os.path.join(Config.WORKING_DIR, "model_engine.log")
)


class EnsembleModel:
    """
    Encapsulates a Bagging Ensemble of Logistic Regression classifiers.
    Includes logic for hyperparameter tuning via Grid Search and model persistence.
    """

    def __init__(self):
        """
        Initializes the EnsembleModel.
        """
        self.model = None
        self.best_params = None

    def optimize_and_train(self, X_train, y_train):
        """
        Performs Grid Search to find the best hyperparameters for the base estimator
        within the context of the Bagging Ensemble, then trains the final model.

        Args:
            X_train (np.ndarray): Training feature matrix.
            y_train (np.ndarray): Training target labels.
        """
        logger.info("Initializing Base Estimator (Logistic Regression)...")
        # Base estimator: Logistic Regression with Ridge (L2) penalty (default)
        # Note: Solver and penalty are part of the grid or defaults
        base_estimator = LogisticRegression(random_state=Config.SEED)

        logger.info(
            f"Initializing Bagging Classifier with {Config.BAGGING_N_ESTIMATORS} estimators..."
        )
        # Bagging Wrapper
        # We set n_jobs=1 here and use n_jobs=-1 in GridSearchCV to parallelize the grid search
        bagging_clf = BaggingClassifier(
            estimator=base_estimator,
            n_estimators=Config.BAGGING_N_ESTIMATORS,
            random_state=Config.SEED,
            n_jobs=1,
        )

        # Prepare Parameter Grid
        # Scikit-learn >= 1.2 uses 'estimator' parameter in BaggingClassifier.
        # We must prefix the base estimator's params with 'estimator__'
        param_grid = {f"estimator__{k}": v for k, v in Config.PARAM_GRID.items()}

        logger.info(f"Starting Grid Search with parameter grid: {param_grid}")

        # Configure Grid Search
        # cv=3 is chosen for internal validation efficiency
        grid_search = GridSearchCV(
            estimator=bagging_clf,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=3,
            n_jobs=-1,
            verbose=0,
        )

        # Run Grid Search
        grid_search.fit(X_train, y_train)

        # Store results
        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_

        logger.info(f"Grid Search Complete.")
        logger.info(f"Best Parameters: {self.best_params}")
        logger.info(f"Best Internal CV ROC AUC: {grid_search.best_score_:.8f}")

    def predict_proba(self, X):
        """
        Predicts class probabilities for the input data.

        Args:
            X (np.ndarray): Input feature matrix.

        Returns:
            np.ndarray: Probabilities for the positive class (class 1).
        """
        if self.model is None:
            raise RuntimeError(
                "Model has not been trained yet. Call optimize_and_train first."
            )

        # BaggingClassifier.predict_proba averages the probabilities of the base estimators
        probas = self.model.predict_proba(X)

        # Return probability of class 1
        return probas[:, 1]

    def save(self, path: str):
        """
        Saves the trained model to disk.

        Args:
            path (str): Destination file path.
        """
        if self.model is None:
            logger.warning("Attempting to save an untrained model.")

        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """
        Loads a trained model from disk.

        Args:
            path (str): Source file path.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")

        self.model = joblib.load(path)
        logger.info(f"Model loaded from {path}")
