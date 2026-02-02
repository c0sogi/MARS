import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegressionCV


def get_linear_lda(random_state=42):
    """
    Constructs the Expert A: Linear Generative Anchor.

    Architecture:
        StandardScaler -> Linear Discriminant Analysis (LDA) with Ledoit-Wolf shrinkage.
        Uses 'eigen' solver for exact computation on small datasets.

    Args:
        random_state (int): Seed for reproducibility.

    Returns:
        sklearn.pipeline.Pipeline: The constructed pipeline.
    """
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lda", LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto")),
        ]
    )
    return pipeline


def get_discriminative_lr(random_state=42, cv=5):
    """
    Constructs the Expert C: Discriminative Linear Expert.

    Architecture:
        StandardScaler -> LogisticRegressionCV.

    Args:
        random_state (int): Seed for reproducibility.
        cv (int): Number of cross-validation folds.

    Returns:
        sklearn.pipeline.Pipeline: The constructed pipeline.
    """
    # Cite Lesson 00047: Dense logarithmic grid for C to avoid selection bias
    cs_grid = np.logspace(-4, 4, 100)

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr_cv",
                LogisticRegressionCV(
                    Cs=cs_grid,
                    cv=cv,
                    scoring="neg_log_loss",
                    max_iter=5000,  # Cite Lesson 00010: Increased max_iter for convergence
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return pipeline
