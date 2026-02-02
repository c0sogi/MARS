import warnings
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from library.config import Config
from library.utils import setup_logger, set_seed


class TunedLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    A wrapper around LogisticRegression that performs GridSearchCV
    to optimize hyperparameters (C, class_weight, etc.).

    This class is designed to be used as both the Text Expert (Stage 1)
    and the Meta-Learner (Stage 2) in the stacking architecture.
    """

    def __init__(
        self,
        param_grid,
        cv=Config.N_FOLDS,
        n_jobs=Config.N_JOBS,
        random_state=Config.RANDOM_SEED,
        scoring="roc_auc",
    ):
        """
        Args:
            param_grid (dict): Dictionary with parameters names (str) as keys
                               and lists of parameter settings to try as values.
            cv (int): Number of folds for StratifiedKFold.
            n_jobs (int): Number of jobs to run in parallel.
            random_state (int): Seed for reproducibility.
            scoring (str): Scoring metric for GridSearchCV.
        """
        self.param_grid = param_grid
        self.cv = cv
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.scoring = scoring

        self.best_estimator_ = None
        self.best_params_ = None
        self.best_score_ = None
        self.logger = setup_logger("TunedLogisticRegression")

    def fit(self, X, y):
        """
        Runs GridSearchCV to find the best hyperparameters and fits the model.

        Args:
            X (array-like): Feature matrix.
            y (array-like): Target vector.

        Returns:
            self
        """
        # Ensure reproducibility
        set_seed(self.random_state)

        # Suppress convergence warnings for cleaner output during GridSearch
        warnings.filterwarnings("ignore")

        self.logger.info("Starting GridSearchCV for LogisticRegression...")

        # Initialize base model
        # Note: If random_state is in param_grid, it will be overridden by the grid.
        # If not, we set it here to ensure the base estimator is deterministic.
        base_model = LogisticRegression(random_state=self.random_state)

        # Define CV strategy
        cv_strategy = StratifiedKFold(
            n_splits=self.cv, shuffle=True, random_state=self.random_state
        )

        # Configure GridSearchCV
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=self.param_grid,
            scoring=self.scoring,
            cv=cv_strategy,
            n_jobs=self.n_jobs,
            verbose=0,  # Silent execution
            refit=True,
        )

        # Execute Search
        grid_search.fit(X, y)

        # Store results
        self.best_estimator_ = grid_search.best_estimator_
        self.best_params_ = grid_search.best_params_
        self.best_score_ = grid_search.best_score_

        # Log results with full precision
        self.logger.info(
            f"GridSearch complete. Best Score ({self.scoring}): {self.best_score_}"
        )
        self.logger.info(f"Best Parameters: {self.best_params_}")

        return self

    def predict(self, X):
        """
        Predict class labels using the best estimator.
        """
        if self.best_estimator_ is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        """
        Predict class probabilities using the best estimator.
        """
        if self.best_estimator_ is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        return self.best_estimator_.predict_proba(X)
