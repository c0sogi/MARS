import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV


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


def build_tuned_lda(random_state=42):
    """
    Constructs a Tuned Generative Expert pipeline.

    Architecture:
        StandardScaler -> LinearDiscriminantAnalysis (GridSearch on Shrinkage)

    Args:
        random_state (int): Seed for reproducibility.

    Returns:
        sklearn.model_selection.GridSearchCV: The tuning object.
    """
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lda", LinearDiscriminantAnalysis(solver="lsqr")),
        ]
    )

    # Search grid for shrinkage: 'auto' (Ledoit-Wolf) vs fixed values
    # Dense grid to find optimum for Log Loss
    params = {"lda__shrinkage": ["auto"] + list(np.linspace(0.0, 1.0, 21))}

    grid = GridSearchCV(
        pipeline, param_grid=params, cv=5, scoring="neg_log_loss", n_jobs=-1
    )
    return grid


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
                    Cs=20,  # Reduced grid density to prevent overfitting
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
