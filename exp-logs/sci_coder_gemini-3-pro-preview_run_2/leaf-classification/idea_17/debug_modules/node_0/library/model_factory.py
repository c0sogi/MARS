import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegressionCV
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem


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
            ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]
    )
    return pipeline


def get_kernel_lda(random_state=42, pca_variance=0.95, n_nystroem=300):
    """
    Constructs the Expert B: Kernel Generative Expert.

    Architecture:
        StandardScaler -> PCA (Densification) -> Nystroem (RBF Kernel) -> LDA.

    Args:
        random_state (int): Seed for reproducibility.
        pca_variance (float): Variance to retain in PCA step.
        n_nystroem (int): Number of components for Nystroem approximation.

    Returns:
        sklearn.pipeline.Pipeline: The constructed pipeline.
    """
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=pca_variance, random_state=random_state)),
            (
                "nystroem",
                Nystroem(
                    kernel="rbf", n_components=n_nystroem, random_state=random_state
                ),
            ),
            ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
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
    # Dense logarithmic grid for C as specified in the idea
    cs_grid = np.logspace(-4, 4, 20)

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr_cv",
                LogisticRegressionCV(
                    Cs=cs_grid,
                    cv=cv,
                    scoring="neg_log_loss",
                    max_iter=2000,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return pipeline
