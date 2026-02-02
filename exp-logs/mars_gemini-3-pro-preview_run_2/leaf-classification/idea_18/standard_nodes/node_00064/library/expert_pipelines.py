import numpy as np
from functools import partial
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression


def build_global_lda(random_state=42):
    """
    Constructs the Global Generative Anchor pipeline.

    Architecture:
        StandardScaler -> LinearDiscriminantAnalysis (with shrinkage)

    Args:
        random_state (int): Seed for reproducibility (though LDA with lsqr is deterministic given data).

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


def build_denoised_lda(k_features, random_state=42):
    """
    Constructs a Denoised Generative Expert pipeline.

    Architecture:
        StandardScaler -> SelectKBest (Mutual Info) -> LinearDiscriminantAnalysis (with shrinkage)

    Args:
        k_features (int): Number of top features to select.
        random_state (int): Seed for mutual information calculation.

    Returns:
        sklearn.pipeline.Pipeline: The constructed pipeline.
    """
    # Fix random_state for mutual_info_classif to ensure reproducible feature selection
    score_func = partial(mutual_info_classif, random_state=random_state)

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("selector", SelectKBest(score_func=score_func, k=k_features)),
            ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]
    )
    return pipeline


def build_global_lr(random_state=42):
    """
    Constructs the Discriminative Backup pipeline with internal CV for hyperparameter tuning.

    Architecture:
        StandardScaler -> LogisticRegressionCV

    Args:
        random_state (int): Seed for reproducibility.

    Returns:
        sklearn.pipeline.Pipeline: The constructed pipeline.
    """
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr_cv",
                LogisticRegressionCV(
                    Cs=100,  # Dense grid
                    cv=5,  # Internal 5-fold CV
                    scoring="neg_log_loss",  # Explicitly optimize log loss
                    solver="lbfgs",  # Standard solver for multiclass
                    multi_class="multinomial",
                    max_iter=2000,  # Increased iterations for convergence
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )
    return pipeline


def build_fixed_lr(C, random_state=42):
    """
    Constructs a fixed Logistic Regression pipeline using a pre-determined C value.
    Used for the Final Retraining phase to prevent configuration drift.

    Architecture:
        StandardScaler -> LogisticRegression (Fixed C)

    Args:
        C (float): Inverse regularization strength.
        random_state (int): Seed for reproducibility.

    Returns:
        sklearn.pipeline.Pipeline: The constructed pipeline.
    """
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    C=C,
                    solver="lbfgs",
                    multi_class="multinomial",
                    max_iter=2000,
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )
    return pipeline
