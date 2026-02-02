import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from library.config import (
    RANDOM_SEED,
    LR_SOLVER,
    LR_PENALTY,
    LR_MAX_ITER,
    LR_CV_FOLDS,
    LR_CS_GRID,
    LDA_SOLVER,
    LDA_SHRINKAGE,
)


def build_linear_species_model(
    max_iter=LR_MAX_ITER,
    cs_grid=None,
    cv_folds=LR_CV_FOLDS,
    solver=LR_SOLVER,
    random_state=RANDOM_SEED,
):
    """
    Builds the Discriminative Linear Branch model using LogisticRegressionCV.

    Args:
        max_iter (int): Maximum number of iterations for the solver.
        cs_grid (list or int): List of floats for inverse regularization strengths or int for number of grid points.
        cv_folds (int): Number of cross-validation folds.
        solver (str): Algorithm to use in the optimization problem.
        random_state (int): Seed for reproducibility.

    Returns:
        sklearn.linear_model.LogisticRegressionCV: The unfitted linear species model.
    """
    if cs_grid is None:
        cs_grid = LR_CS_GRID

    model = LogisticRegressionCV(
        Cs=cs_grid,
        cv=cv_folds,
        penalty=LR_PENALTY,
        solver=solver,
        max_iter=max_iter,
        random_state=random_state,
        n_jobs=-1,
        multi_class="multinomial",
    )
    return model


def build_generative_species_model(solver=LDA_SOLVER, shrinkage=LDA_SHRINKAGE):
    """
    Builds the Generative Linear Branch model using LinearDiscriminantAnalysis with shrinkage.

    Args:
        solver (str): Solver to use ('lsqr' or 'eigen' required for shrinkage).
        shrinkage (str or float): Shrinkage parameter ('auto' for Ledoit-Wolf).

    Returns:
        sklearn.discriminant_analysis.LinearDiscriminantAnalysis: The unfitted generative model.
    """
    model = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)
    return model
