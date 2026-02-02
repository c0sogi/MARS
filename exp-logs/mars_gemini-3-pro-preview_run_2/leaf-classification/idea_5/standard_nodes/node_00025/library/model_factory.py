import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegressionCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from library.utils import set_seed


def build_linear_pipeline(random_state=42, n_jobs=-1):
    """
    Constructs the Discriminative Linear Component:
    StandardScaler -> LogisticRegressionCV (L2 penalty)

    Args:
        random_state (int): Seed for reproducibility.
        n_jobs (int): Number of CPU cores to use.

    Returns:
        sklearn.pipeline.Pipeline: The constructed linear pipeline.
    """
    set_seed(random_state)

    # Constrain the search space to the relevant regime (0.01 to 10000)
    # This avoids searching extremely high regularization areas that underfit
    Cs = np.logspace(-2, 4, 20)

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegressionCV(
                    Cs=Cs,
                    cv=3,
                    penalty="l2",
                    solver="lbfgs",
                    multi_class="multinomial",
                    max_iter=5000,  # Increased to ensure convergence
                    scoring="neg_log_loss",
                    random_state=random_state,
                    n_jobs=n_jobs,
                ),
            ),
        ]
    )
    return pipeline


def build_generative_pipeline():
    """
    Constructs the Generative Linear Component:
    StandardScaler -> LinearDiscriminantAnalysis (Ledoit-Wolf shrinkage)

    Returns:
        sklearn.pipeline.Pipeline: The constructed generative pipeline.
    """
    # LDA with Ledoit-Wolf shrinkage is deterministic given the data,
    # but we ensure the environment is seeded via the caller if needed.

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LinearDiscriminantAnalysis(
                    solver="lsqr", shrinkage="auto"  # Implements Ledoit-Wolf shrinkage
                ),
            ),
        ]
    )
    return pipeline
