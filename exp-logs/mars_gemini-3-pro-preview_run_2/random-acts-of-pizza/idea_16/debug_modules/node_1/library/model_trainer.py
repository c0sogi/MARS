import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import set_seed, print_metric


class ModelTrainer:
    """
    Manages the creation, optimization, and training of the Bagged ElasticNet Ensemble.
    """

    def __init__(self):
        """
        Initialize the trainer.
        """
        set_seed(Config.SEED)

    def build_pipeline(self, params=None):
        """
        Constructs the base Logistic Regression estimator with ElasticNet penalty.

        Args:
            params (dict, optional): Hyperparameters to override defaults.

        Returns:
            sklearn.linear_model.LogisticRegression: The configured base estimator.
        """
        # Start with default base estimator params from Config
        model_params = Config.BASE_ESTIMATOR_PARAMS.copy()

        # Update with any provided specific parameters (e.g., from GridSearch)
        if params:
            model_params.update(params)

        # Initialize Logistic Regression with SAGA solver for ElasticNet support
        model = LogisticRegression(**model_params)

        return model

    def optimize_and_train(self, X_train, y_train):
        """
        Performs hyperparameter tuning using GridSearchCV and then trains a
        BaggingClassifier using the best found parameters.

        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training labels.

        Returns:
            tuple: (trained_bagging_model, best_params)
        """
        set_seed(Config.SEED)

        print("Starting Grid Search for Logistic Regression hyperparameters...")

        # 1. Hyperparameter Tuning (Grid Search)
        # We use a temporary base estimator for tuning
        base_estimator = self.build_pipeline()

        # Configure GridSearchCV
        # n_jobs=-1 uses all available processors
        grid_search = GridSearchCV(
            estimator=base_estimator,
            param_grid=Config.PARAM_GRID,
            scoring="roc_auc",
            cv=5,  # Internal 5-fold CV for hyperparameter validation
            n_jobs=-1,
            verbose=0,  # Keep output clean as requested
            refit=False,  # We will manually refit inside the BaggingClassifier
        )

        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        print("Grid Search completed.")
        print(f"Best Parameters: {best_params}")
        print_metric("Best Internal CV ROC AUC", best_score)

        # 2. Ensemble Training (Bagging)
        print("Training Bagged Ensemble with best parameters...")

        # Create the base estimator with the optimal parameters
        optimized_base_estimator = self.build_pipeline(best_params)

        # Initialize the BaggingClassifier
        bagging_clf = BaggingClassifier(
            estimator=optimized_base_estimator, **Config.BAGGING_PARAMS
        )

        # Fit the ensemble on the training data
        bagging_clf.fit(X_train, y_train)

        return bagging_clf, best_params
