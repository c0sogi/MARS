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
    # Cite solution_lesson_node_00047: Prioritize grid density and range.
    # Shifted to -3 to 5 to cover weaker regularization (Higher C) for high-signal features.
    Cs = np.logspace(-3, 5, 100)

    clf = LogisticRegressionCV(
        Cs=Cs,
        cv=3,
        scoring="neg_log_loss",
        solver="lbfgs",
        penalty="l2",
        multi_class="multinomial",
        max_iter=5000,  # Cite solution_lesson_node_00010: Increased to ensure convergence
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
