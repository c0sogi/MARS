from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegressionCV
from sklearn.covariance import OAS
from library.config import Config


def get_expert_model(model_type):
    """
    Factory function to instantiate the specific estimator for an expert.

    Supported Models:
    1. LDA with OAS (Oracle Approximating Shrinkage):
       - Uses the OAS algorithm to robustly estimate the covariance matrix.
       - Solver: 'lsqr' (required for custom covariance estimators).

    2. LDA with Fixed Shrinkage:
       - Uses a fixed scalar shrinkage coefficient defined in Config.
       - Solver: 'lsqr' (required for shrinkage).

    3. Logistic Regression CV:
       - Discriminative anchor using L2 regularization.
       - Automatically tunes C via Cross-Validation optimizing neg_log_loss.

    Args:
        model_type (str): Identifier for the model type (from Config).

    Returns:
        BaseEstimator: An instantiated scikit-learn estimator.

    Raises:
        ValueError: If model_type is not recognized.
    """
    if model_type == Config.MODEL_LDA_OAS:
        # Linear Discriminant Analysis with OAS Covariance Estimator
        # OAS is generally superior to Ledoit-Wolf for Gaussian-like data with N < p or N ~ p
        return LinearDiscriminantAnalysis(
            solver="lsqr",
            covariance_estimator=OAS(),
            store_covariance=True,  # Useful for debugging or analysis if needed
        )

    elif model_type == Config.MODEL_LDA_FIXED:
        # Linear Discriminant Analysis with Fixed Shrinkage
        # Applies a fixed amount of regularization to the covariance matrix
        return LinearDiscriminantAnalysis(
            solver="lsqr", shrinkage=Config.LDA_FIXED_SHRINKAGE, store_covariance=True
        )

    elif model_type == Config.MODEL_LOGREG:
        # Logistic Regression with Cross-Validation
        # Serves as the discriminative fallback/anchor
        return LogisticRegressionCV(
            cv=5,  # 5-fold CV for hyperparameter tuning
            penalty="l2",
            solver="lbfgs",
            scoring="neg_log_loss",
            max_iter=Config.LOGREG_MAX_ITER,
            random_state=Config.RANDOM_SEED,
            n_jobs=-1,  # Use all available cores
            multi_class="multinomial",  # Explicitly handle multi-class
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")
