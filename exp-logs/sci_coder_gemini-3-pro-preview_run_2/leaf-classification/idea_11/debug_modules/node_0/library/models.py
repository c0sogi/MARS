import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import Pipeline


def get_linear_branch(cv=3, random_state=42, max_iter=2000, n_jobs=-1):
    """
    Returns the Discriminative Linear Branch: A Logistic Regression model.

    Configuration:
    - LogisticRegressionCV with L2 regularization.
    - Broad search grid for Cs (1e-2 to 1e4).
    - Optimizes neg_log_loss.
    - Uses 'lbfgs' solver for robust convergence.
    """
    # Broad logarithmic grid for Cs to ensure global optimum is found
    Cs = np.logspace(-2, 4, 20)

    clf = LogisticRegressionCV(
        Cs=Cs,
        cv=cv,
        penalty="l2",
        solver="lbfgs",
        scoring="neg_log_loss",
        multi_class="multinomial",
        max_iter=max_iter,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    return clf


def get_generative_branch(solver="lsqr", shrinkage="ledoit_wolf"):
    """
    Returns the Generative Linear Branch: A Linear Discriminant Analysis (LDA) model.

    Configuration:
    - Uses Ledoit-Wolf shrinkage to handle high dimensionality and small sample size.
    - Solver set to 'lsqr' to support shrinkage.
    """
    clf = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)
    return clf


def get_kernel_branch(
    cv=3,
    random_state=42,
    max_iter=2000,
    n_jobs=-1,
    pca_variance=0.95,
    nystroem_components=300,
):
    """
    Returns the Discriminative Kernel Branch: A Nystroem Kernel Logistic Regression pipeline.

    Architecture:
    1. PCA (retaining 95% variance) -> Densifies space, removes noise.
    2. Nystroem (RBF kernel approximation) -> Projects to non-linear manifold.
    3. LogisticRegressionCV -> Robust linear solver on non-linear features.
    """
    # 1. PCA to densify feature space and remove noise
    # n_components < 1.0 implies variance retention
    pca = PCA(n_components=pca_variance, random_state=random_state)

    # 2. Nystroem Kernel Approximation (RBF)
    nystroem = Nystroem(
        kernel="rbf", n_components=nystroem_components, random_state=random_state
    )

    # 3. Logistic Regression Classifier (Linear solver on non-linear features)
    # Re-using the robust configuration from the linear branch
    Cs = np.logspace(-2, 4, 20)
    clf = LogisticRegressionCV(
        Cs=Cs,
        cv=cv,
        penalty="l2",
        solver="lbfgs",
        scoring="neg_log_loss",
        multi_class="multinomial",
        max_iter=max_iter,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    pipeline = Pipeline([("pca", pca), ("nystroem", nystroem), ("classifier", clf)])

    return pipeline
