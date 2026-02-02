import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from library.config import (
    LR_SOLVER,
    LR_PENALTY,
    LR_MAX_ITER,
    LR_C_GRID,
    CV_FOLDS,
    LDA_SOLVER,
    LDA_SHRINKAGE,
    RANDOM_SEED,
)


def get_probabilistic_ensemble():
    """
    Constructs the LDA and LogisticRegressionCV models for the ensemble.

    Returns:
        tuple: (lda_model, lr_model)
            - lda_model: The LinearDiscriminantAnalysis instance (Generative Branch).
            - lr_model: The LogisticRegressionCV instance (Discriminative Branch).
    """
    print("Initializing Probabilistic Ensemble components...")

    # 1. Generative Branch: Linear Discriminant Analysis (Anchor)
    # Uses Ledoit-Wolf shrinkage to handle high-dimensional, low-sample data effectively.
    lda_model = LinearDiscriminantAnalysis(solver=LDA_SOLVER, shrinkage=LDA_SHRINKAGE)

    # 2. Discriminative Branch: Logistic Regression CV
    # Uses integrated cross-validation to find the optimal C and refit the model.
    # This avoids manual tune-and-refit loops and feature subsampling.
    lr_model = LogisticRegressionCV(
        Cs=LR_C_GRID,
        cv=CV_FOLDS,
        solver=LR_SOLVER,
        penalty=LR_PENALTY,
        max_iter=LR_MAX_ITER,
        multi_class="multinomial",
        scoring="neg_log_loss",
        n_jobs=-1,
        random_state=RANDOM_SEED,
        verbose=0,
        refit=True,
    )

    return lda_model, lr_model
