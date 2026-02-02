import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.linear_model import LogisticRegressionCV
from library.config import Config


def build_global_lda():
    """
    Constructs the Global Generative Anchor expert.

    Architecture:
        StandardScaler -> Linear Discriminant Analysis (LDA)

    Configuration:
        - Solver: LSQR (supports shrinkage)
        - Shrinkage: Auto (Ledoit-Wolf lemma)

    Returns:
        sklearn.pipeline.Pipeline: The initialized pipeline.
    """
    steps = [
        ("scaler", StandardScaler()),
        (
            "lda",
            LinearDiscriminantAnalysis(
                solver=Config.LDA_SOLVER, shrinkage=Config.LDA_SHRINKAGE
            ),
        ),
    ]
    return Pipeline(steps)


def build_metric_lda():
    """
    Constructs the Metric-Optimized Generative Expert.

    Architecture:
        StandardScaler -> Neighborhood Components Analysis (NCA) -> LDA

    Rationale:
        NCA learns a linear transformation that maximizes stochastic nearest-neighbor
        accuracy, effectively handling the 'Curse of Dimensionality' before the
        generative LDA step.

    Returns:
        sklearn.pipeline.Pipeline: The initialized pipeline.
    """
    steps = [
        ("scaler", StandardScaler()),
        (
            "nca",
            NeighborhoodComponentsAnalysis(
                n_components=Config.NCA_COMPONENTS,
                init=Config.NCA_INIT,
                max_iter=Config.NCA_MAX_ITER,
                tol=Config.NCA_TOL,
                random_state=Config.RANDOM_SEED,
            ),
        ),
        (
            "lda",
            LinearDiscriminantAnalysis(
                solver=Config.LDA_SOLVER, shrinkage=Config.LDA_SHRINKAGE
            ),
        ),
    ]
    return Pipeline(steps)


def build_discriminative_lr():
    """
    Constructs the Discriminative Linear Expert.

    Architecture:
        StandardScaler -> Logistic Regression (with CV)

    Configuration:
        - Optimization: LBFGS
        - Grid Search: Dense logarithmic grid for C (Inverse Regularization)
        - Metric: Negative Log Loss

    Returns:
        sklearn.pipeline.Pipeline: The initialized pipeline.
    """
    # Note: LogisticRegressionCV automatically handles the grid search efficiently.
    # We use 'multinomial' implicitly by the nature of the data/solver,
    # but 'neg_log_loss' scoring ensures we optimize the competition metric.
    steps = [
        ("scaler", StandardScaler()),
        (
            "logreg",
            LogisticRegressionCV(
                Cs=Config.LOGREG_C_GRID,
                cv=5,  # Standard 5-fold internal CV
                solver=Config.LOGREG_SOLVER,
                max_iter=Config.LOGREG_MAX_ITER,
                scoring="neg_log_loss",
                n_jobs=Config.LOGREG_JOBS,
                random_state=Config.RANDOM_SEED,
                refit=True,
            ),
        ),
    ]
    return Pipeline(steps)


def get_expert_pool():
    """
    Factory function to instantiate the pool of experts for the ensemble.

    Returns:
        dict: A dictionary where keys are expert names and values are
              untrained sklearn Pipelines.
    """
    return {
        "Global_LDA": build_global_lda(),
        "Metric_LDA": build_metric_lda(),
        "Discriminative_LR": build_discriminative_lr(),
    }
