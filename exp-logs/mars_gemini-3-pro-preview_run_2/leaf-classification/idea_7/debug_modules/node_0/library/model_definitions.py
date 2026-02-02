import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from library.config import (
    RANDOM_SEED,
    PCA_VARIANCE,
    LR_SOLVER,
    LR_PENALTY,
    LR_MAX_ITER,
    LR_CV_FOLDS,
    LR_CS_GRID,
    LDA_SOLVER,
    LDA_SHRINKAGE,
    POLY_DEGREE,
    POLY_INTERACTION_ONLY,
    POLY_INCLUDE_BIAS,
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


def build_quadratic_species_model(
    pca_variance=PCA_VARIANCE,
    poly_degree=POLY_DEGREE,
    max_iter=LR_MAX_ITER,
    cs_grid=None,
    cv_folds=LR_CV_FOLDS,
    random_state=RANDOM_SEED,
):
    """
    Builds the Discriminative Quadratic Branch pipeline.
    Pipeline: PCA -> PolynomialFeatures -> LogisticRegressionCV.

    Args:
        pca_variance (float): Amount of variance to retain in PCA (0.0 to 1.0).
        poly_degree (int): The degree of the polynomial features.
        max_iter (int): Maximum iterations for the logistic regression.
        cs_grid (list): Regularization grid for logistic regression.
        cv_folds (int): Cross-validation folds.
        random_state (int): Seed for reproducibility.

    Returns:
        sklearn.pipeline.Pipeline: The unfitted quadratic pipeline.
    """
    if cs_grid is None:
        cs_grid = LR_CS_GRID

    # 1. Dimensionality Reduction
    pca = PCA(n_components=pca_variance, random_state=random_state)

    # 2. Feature Interaction
    poly = PolynomialFeatures(
        degree=poly_degree,
        interaction_only=POLY_INTERACTION_ONLY,
        include_bias=POLY_INCLUDE_BIAS,
    )

    # 3. Classifier
    clf = LogisticRegressionCV(
        Cs=cs_grid,
        cv=cv_folds,
        penalty=LR_PENALTY,
        solver=LR_SOLVER,
        max_iter=max_iter,
        random_state=random_state,
        n_jobs=-1,
        multi_class="multinomial",
    )

    pipeline = Pipeline([("pca", pca), ("poly", poly), ("clf", clf)])

    return pipeline


def build_genus_supervisor_model(
    max_iter=LR_MAX_ITER, cs_grid=None, cv_folds=LR_CV_FOLDS, random_state=RANDOM_SEED
):
    """
    Builds the Genus-Level Supervisor model.
    Structurally similar to the linear species model but trained on genus targets.

    Args:
        max_iter (int): Maximum number of iterations.
        cs_grid (list): Regularization grid.
        cv_folds (int): CV folds.
        random_state (int): Seed for reproducibility.

    Returns:
        sklearn.linear_model.LogisticRegressionCV: The unfitted genus supervisor model.
    """
    if cs_grid is None:
        cs_grid = LR_CS_GRID

    model = LogisticRegressionCV(
        Cs=cs_grid,
        cv=cv_folds,
        penalty=LR_PENALTY,
        solver=LR_SOLVER,
        max_iter=max_iter,
        random_state=random_state,
        n_jobs=-1,
        multi_class="multinomial",
    )
    return model
