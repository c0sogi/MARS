from sklearn.linear_model import RidgeCV
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV
from library.config import Config


class ModelFactory:
    """
    Factory class to instantiate Level-0 and Level-1 estimators
    with hyperparameters defined in Config.
    """

    @staticmethod
    def get_linear_expert(random_state=Config.SEED):
        """
        Returns the Linear Expert (Ridge Regression).
        Uses RidgeCV for efficient Leave-One-Out Cross-Validation (LOOCV/GCV).

        Args:
            random_state (int): Seed for reproducibility (unused by RidgeCV as it is deterministic).

        Returns:
            sklearn.linear_model.RidgeCV: The initialized linear model.
        """
        # RidgeCV uses Generalized Cross-Validation (GCV) by default which is very efficient.
        # We pass the alphas from config.
        return RidgeCV(alphas=Config.RIDGE_ALPHAS)

    @staticmethod
    def get_kernel_expert(random_state=Config.SEED):
        """
        Returns the Kernel Expert (Support Vector Regression).
        Wraps SVR in GridSearchCV to optimize 'C' if multiple values are provided.

        Args:
            random_state (int): Seed for reproducibility (unused by SVR).

        Returns:
            sklearn.svm.SVR or sklearn.model_selection.GridSearchCV: The initialized kernel model.
        """
        svr = SVR(kernel=Config.SVR_KERNEL, epsilon=Config.SVR_EPSILON)

        # Check if we need to search for C
        c_values = Config.SVR_C
        if isinstance(c_values, list) and len(c_values) > 1:
            param_grid = {"C": c_values}
            # Use 3-fold CV for speed during grid search, as SVR is computationally expensive
            return GridSearchCV(
                estimator=svr,
                param_grid=param_grid,
                scoring="neg_root_mean_squared_error",
                cv=3,
                n_jobs=Config.NUM_WORKERS,
            )
        else:
            # Use the single provided C value
            if isinstance(c_values, list):
                svr.C = c_values[0]
            else:
                svr.C = c_values
            return svr

    @staticmethod
    def get_partitioning_expert(random_state=Config.SEED):
        """
        Returns the Partitioning Expert (ExtraTrees Regressor).
        Configured to run in parallel using Config.NUM_WORKERS.

        Args:
            random_state (int): Seed for reproducibility.

        Returns:
            sklearn.ensemble.ExtraTreesRegressor: The initialized tree-based model.
        """
        return ExtraTreesRegressor(
            n_estimators=Config.ET_N_ESTIMATORS,
            min_samples_split=Config.ET_MIN_SAMPLES_SPLIT,
            random_state=random_state,
            n_jobs=Config.NUM_WORKERS,
            verbose=0,
        )

    @staticmethod
    def get_meta_learner(random_state=Config.SEED):
        """
        Returns the Level-1 Meta-Learner (Ridge Regression).
        Aggregates predictions from Level-0 experts.

        Args:
            random_state (int): Seed for reproducibility (unused by RidgeCV).

        Returns:
            sklearn.linear_model.RidgeCV: The initialized meta-learner.
        """
        return RidgeCV(alphas=Config.META_RIDGE_ALPHAS)
