import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from library.utils import set_seed


def build_linear_branch(random_state=42, n_jobs=-1):
    """
    Constructs the Discriminative Linear Branch:
    StandardScaler -> LogisticRegressionCV

    Features:
    - Dense, Broad Logarithmic Grid (100 points) for C.
    - Optimized for neg_log_loss.
    - L2 Regularization.
    """
    # Dense, Broad Logarithmic Grid (100 points)
    # Range 1e-4 to 1e4 covers most practical regularization strengths
    Cs = np.logspace(-4, 4, 100)

    clf = LogisticRegressionCV(
        Cs=Cs,
        cv=3,
        scoring="neg_log_loss",
        solver="lbfgs",
        penalty="l2",
        multi_class="multinomial",
        max_iter=2000,  # Increased to ensure convergence
        random_state=random_state,
        n_jobs=n_jobs,
    )

    pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", clf)])

    return pipeline


def build_generative_branch():
    """
    Constructs the Generative Linear Branch:
    StandardScaler -> LDA with Ledoit-Wolf shrinkage

    Features:
    - Robust covariance estimation using Ledoit-Wolf shrinkage.
    - Sample efficient.
    """
    # LDA with Ledoit-Wolf shrinkage requires lsqr or eigen solver
    clf = LinearDiscriminantAnalysis(
        solver="lsqr", shrinkage="auto"  # 'auto' results in Ledoit-Wolf lemma
    )

    pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", clf)])

    return pipeline
