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

    Args:
        random_state (int): Seed for reproducibility (not used by LDA lsqr solver directly,
                            but included for interface consistency).

    Returns:
        sklearn.pipeline.Pipeline: The constructed pipeline.
    """
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lda",
                LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", tol=1e-8),
            ),
        ]
    )
    return pipeline


def get_discriminative_lr(random_state=42, cv=3):
    """
    Constructs the Expert C: Discriminative Linear Expert.

    Architecture:
        StandardScaler -> LogisticRegressionCV.

    Args:
        random_state (int): Seed for reproducibility.
        cv (int): Number of cross-validation folds. Reduced to 3 for small data (Cite 00010).

    Returns:
        sklearn.pipeline.Pipeline: The constructed pipeline.
    """
    # Dense logarithmic grid for C (Cite 00047, 00037)
    cs_grid = np.logspace(-5, 5, 100)

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr_cv",
                LogisticRegressionCV(
                    Cs=cs_grid,
                    cv=cv,
                    scoring="neg_log_loss",
                    max_iter=5000,  # Increased max_iter (Cite 00010)
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return pipeline
