import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from library.config import Config
from library.utils import setup_logger


class ModelFactory:
    """
    Factory class for creating and optimizing the Coherence-Augmented
    Bagged Logistic Regression Ensemble.
    """

    def __init__(self):
        self.logger = setup_logger(
            "ModelFactory", "./working/idea_40/model_factory.log"
        )

    def optimize_and_train(self, X_train, y_train):
        """
        Performs GridSearchCV to find the best hyperparameters for the Bagged Logistic Regression,
        then returns the best estimator trained on the provided data.

        Args:
            X_train (array-like): Training features.
            y_train (array-like): Training labels.

        Returns:
            best_estimator_: The fitted BaggingClassifier with optimal parameters.
        """
        self.logger.info(f"Starting model optimization on input shape: {X_train.shape}")

        # Base Estimator: Logistic Regression (Ridge)
        # We set random_state for reproducibility.
        # Note: Solver and max_iter are also part of the grid, but we set defaults here.
        base_estimator = LogisticRegression(random_state=Config.SEED)

        # Bagging Wrapper
        # We set n_jobs=1 here because GridSearchCV will handle parallelization across folds.
        # This avoids nested parallelism which can cause overhead.
        bagging_clf = BaggingClassifier(
            estimator=base_estimator,
            n_estimators=Config.BAGGING_N_ESTIMATORS,
            random_state=Config.SEED,
            n_jobs=1,
        )

        # Grid Search Configuration
        # scoring='roc_auc' automatically handles predict_proba and positive class selection
        grid_search = GridSearchCV(
            estimator=bagging_clf,
            param_grid=Config.PARAM_GRID,
            scoring="roc_auc",
            cv=3,  # 3-fold internal CV for efficiency within the outer fold
            n_jobs=Config.N_JOBS,
            verbose=0,
            refit=True,  # Refit on the whole X_train with best params
        )

        # Execute Grid Search
        grid_search.fit(X_train, y_train)

        best_score = grid_search.best_score_
        best_params = grid_search.best_params_

        # Log results with full precision as requested
        self.logger.info(
            f"Optimization completed. Best Internal CV ROC AUC: {best_score}"
        )
        self.logger.info(f"Best Hyperparameters: {best_params}")

        return grid_search.best_estimator_

    def get_default_model(self):
        """
        Returns a Bagged Ensemble with default parameters.
        Useful for debugging or if optimization is skipped.
        """
        base_estimator = LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs", max_iter=1000, random_state=Config.SEED
        )

        model = BaggingClassifier(
            estimator=base_estimator,
            n_estimators=Config.BAGGING_N_ESTIMATORS,
            random_state=Config.SEED,
            n_jobs=Config.N_JOBS,
        )
        return model
