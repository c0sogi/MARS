import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV
from library.config import Config
from library.utils import setup_logger


class ModelBuilder:
    """
    Constructs the machine learning model architecture.

    Implements a Bagged Ensemble of Logistic Regression Classifiers optimized
    via GridSearchCV, as defined in the Whitened Multi-Field Asymmetric
    Dual-Backbone Ensemble strategy.
    """

    def __init__(self):
        self.logger = setup_logger("ModelBuilder")

    def get_bagged_lr_optimizer(self):
        """
        Constructs and returns a GridSearchCV object for a Bagged Logistic Regression.

        The architecture consists of:
        1. Base Estimator: LogisticRegression (Ridge/L2).
        2. Ensemble: BaggingClassifier (reduces variance of the linear core).
        3. Optimization: GridSearchCV (tunes C and class_weight of the ensemble).

        Returns:
            GridSearchCV: The configured optimizer object, ready for .fit().
        """
        self.logger.info("Building Bagged Logistic Regression Optimizer...")

        # 1. Define the Base Estimator
        # Initial parameters are placeholders; they will be tuned by GridSearch.
        # We fix random_state for reproducibility.
        base_estimator = LogisticRegression(random_state=Config.SEED)

        # 2. Define the Bagging Ensemble
        # We set n_jobs=1 here to avoid oversubscription, as GridSearchCV will
        # parallelize the search process across folds/parameters (n_jobs=-1).
        bagging_clf = BaggingClassifier(
            estimator=base_estimator,
            n_estimators=Config.N_BAGGING_ESTIMATORS,
            random_state=Config.SEED,
            n_jobs=1,
        )

        # 3. Construct the Parameter Grid
        # The keys in Config.LR_PARAM_GRID (e.g., 'C', 'class_weight') apply to the
        # base LogisticRegression. Since it is wrapped in BaggingClassifier, we must
        # prefix keys with 'estimator__' (scikit-learn >= 1.2 convention).
        param_grid = {}
        for param, values in Config.LR_PARAM_GRID.items():
            grid_key = f"estimator__{param}"
            param_grid[grid_key] = values

        self.logger.info(
            f"Ensemble Configuration: n_estimators={Config.N_BAGGING_ESTIMATORS}"
        )
        self.logger.info(f"Search Grid: {param_grid}")

        # 4. Initialize GridSearchCV
        # We use 'roc_auc' as the scoring metric as per the task requirement.
        optimizer = GridSearchCV(
            estimator=bagging_clf,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=Config.N_FOLDS,
            n_jobs=-1,  # Use all available vCPUs
            verbose=1,
            refit=True,  # Refit the best model on the full training data of the fold
        )

        return optimizer
