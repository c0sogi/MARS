import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def get_linear_branch(cv=3, random_state=42, max_iter=5000, n_jobs=-1):
    """
    Returns the Discriminative Linear Branch: A Logistic Regression model.

    Configuration:
    - LogisticRegressionCV with L2 regularization.
    - Broad search grid for Cs (1e-2 to 1e4).
    - Optimizes neg_log_loss.
    - Uses 'lbfgs' solver for robust convergence.
    """
    # Broad logarithmic grid for Cs to ensure global optimum is found
    # Cite solution_lesson_node_00037: Broadening Regularization Search Spaces
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


def get_generative_branch(solver="lsqr", shrinkage="auto"):
    """
    Returns the Generative Linear Branch: A Linear Discriminant Analysis (LDA) model.

    Configuration:
    - Uses Ledoit-Wolf shrinkage to handle high dimensionality and small sample size.
    - Solver set to 'lsqr' to support shrinkage.
    """
    # Cite solution_lesson_node_00006: Generative-Discriminative Ensembling
    clf = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)
    return clf
