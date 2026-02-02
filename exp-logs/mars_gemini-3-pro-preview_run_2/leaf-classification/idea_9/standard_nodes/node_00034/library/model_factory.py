import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem

# Set global random state
RANDOM_STATE = 42


def get_linear_expert(random_state=RANDOM_STATE):
    """
    Returns the Discriminative Linear Expert: StandardScaler + LogisticRegressionCV.

    This model captures the strong linear separability of the feature space.
    """
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegressionCV(
                    Cs=10,
                    cv=3,
                    solver="lbfgs",
                    max_iter=2000,
                    scoring="neg_log_loss",
                    multi_class="multinomial",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return pipeline


def get_generative_expert(random_state=RANDOM_STATE):
    """
    Returns the Generative Linear Expert: StandardScaler + LDA with shrinkage.

    This model provides superior sample efficiency and density estimation
    in the "Small N, High D" regime using Ledoit-Wolf shrinkage.
    """
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]
    )
    return pipeline


def get_kernel_expert(random_state=RANDOM_STATE):
    """
    Returns the Discriminative Non-Linear Expert:
    StandardScaler + PCA (0.95 variance) + Nystroem + LogisticRegressionCV.

    This pipeline approximates the RBF kernel map explicitly to capture
    non-linear manifold structures while retaining the calibration benefits
    of Logistic Regression.
    """
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.95, random_state=random_state)),
            (
                "nystroem",
                Nystroem(
                    kernel="rbf",
                    gamma=None,  # Defaults to 1/n_features
                    n_components=400,  # Project to higher dimension for non-linearity
                    random_state=random_state,
                ),
            ),
            (
                "clf",
                LogisticRegressionCV(
                    Cs=10,
                    cv=3,
                    solver="lbfgs",
                    max_iter=2000,
                    scoring="neg_log_loss",
                    multi_class="multinomial",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return pipeline


def get_meta_learner(random_state=RANDOM_STATE):
    """
    Returns the Level 1 Meta-Learner: Logistic Regression.

    This model takes the Out-of-Fold (OOF) probability predictions from the
    three base learners and learns the optimal mixing weights to minimize Log Loss.
    """
    clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        multi_class="multinomial",
        max_iter=1000,
        random_state=random_state,
        n_jobs=-1,
    )
    return clf
