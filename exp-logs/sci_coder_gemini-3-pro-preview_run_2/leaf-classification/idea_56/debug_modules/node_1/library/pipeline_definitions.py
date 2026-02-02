import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def get_expert_library(feature_groups, shrinkage_levels=[0.001, 0.01]):
    """
    Constructs the library of expert pipelines for the MS-DIPGE ensemble.

    Args:
        feature_groups (dict): Dictionary containing lists of column names for
                               'global', 'morph', 'margin', 'shape', 'texture'.
        shrinkage_levels (list): List of float shrinkage values for the final LDA solver.

    Returns:
        dict: A dictionary where keys are expert names and values are sklearn Pipelines.
    """
    experts = {}

    # Extract column groups
    global_cols = feature_groups["global"]
    morph_cols = feature_groups["morph"]
    margin_cols = feature_groups["margin"]
    shape_cols = feature_groups["shape"]
    texture_cols = feature_groups["texture"]

    # -------------------------------------------------------------------------
    # Helper: Final LDA Solver
    # -------------------------------------------------------------------------
    def make_solver(shrinkage):
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage=shrinkage)

    # -------------------------------------------------------------------------
    # Group A: Global Linear Anchors
    # -------------------------------------------------------------------------
    # A1: Marginal Topology
    # Input: Global -> PT -> LDA
    for s in shrinkage_levels:
        name = f"GroupA_Marginal_s{s}"
        pipeline = Pipeline(
            [
                ("selector", ColumnTransformer([("sel", "passthrough", global_cols)])),
                ("pt", PowerTransformer()),
                ("clf", make_solver(s)),
            ]
        )
        experts[name] = pipeline

    # A2: Rotational Topology
    # Input: Global -> PT -> PCA -> PT -> LDA
    for s in shrinkage_levels:
        name = f"GroupA_Rotational_s{s}"
        pipeline = Pipeline(
            [
                ("selector", ColumnTransformer([("sel", "passthrough", global_cols)])),
                ("pt1", PowerTransformer()),
                ("pca", PCA(whiten=False)),
                ("pt2", PowerTransformer()),
                ("clf", make_solver(s)),
            ]
        )
        experts[name] = pipeline

    # -------------------------------------------------------------------------
    # Group B: Physical Polynomial Experts
    # -------------------------------------------------------------------------
    # Input: Morph -> PT -> Poly(2) -> PT -> LDA
    for s in shrinkage_levels:
        name = f"GroupB_Physical_s{s}"
        pipeline = Pipeline(
            [
                ("selector", ColumnTransformer([("sel", "passthrough", morph_cols)])),
                ("pt1", PowerTransformer()),
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("pt2", PowerTransformer()),
                ("clf", make_solver(s)),
            ]
        )
        experts[name] = pipeline

    # -------------------------------------------------------------------------
    # Group C: Global Discriminative-Interaction Experts
    # -------------------------------------------------------------------------
    # Input: Global -> PT -> LDA_Trans(15) -> Poly(2) -> PT -> LDA
    # Note: Intermediate LDA uses 'svd' for robust projection.
    for s in shrinkage_levels:
        name = f"GroupC_GlobalInteract_s{s}"
        pipeline = Pipeline(
            [
                ("selector", ColumnTransformer([("sel", "passthrough", global_cols)])),
                ("pt1", PowerTransformer()),
                ("lda_proj", LinearDiscriminantAnalysis(n_components=15, solver="svd")),
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("pt2", PowerTransformer()),
                ("clf", make_solver(s)),
            ]
        )
        experts[name] = pipeline

    # -------------------------------------------------------------------------
    # Group D: Stratified Discriminative-Interaction Experts
    # -------------------------------------------------------------------------
    # Input:
    #   Margin -> PT -> LDA_Trans(9) \
    #   Shape  -> PT -> LDA_Trans(9)  -> Concat -> Poly(2) -> PT -> LDA
    #   Texture-> PT -> LDA_Trans(9) /

    # Define the stratified projection transformer
    stratified_projector = ColumnTransformer(
        [
            (
                "margin_proj",
                Pipeline(
                    [
                        ("pt", PowerTransformer()),
                        (
                            "lda",
                            LinearDiscriminantAnalysis(n_components=9, solver="svd"),
                        ),
                    ]
                ),
                margin_cols,
            ),
            (
                "shape_proj",
                Pipeline(
                    [
                        ("pt", PowerTransformer()),
                        (
                            "lda",
                            LinearDiscriminantAnalysis(n_components=9, solver="svd"),
                        ),
                    ]
                ),
                shape_cols,
            ),
            (
                "texture_proj",
                Pipeline(
                    [
                        ("pt", PowerTransformer()),
                        (
                            "lda",
                            LinearDiscriminantAnalysis(n_components=9, solver="svd"),
                        ),
                    ]
                ),
                texture_cols,
            ),
        ]
    )

    for s in shrinkage_levels:
        name = f"GroupD_StratifiedInteract_s{s}"
        pipeline = Pipeline(
            [
                ("stratified_proj", stratified_projector),
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("pt", PowerTransformer()),
                ("clf", make_solver(s)),
            ]
        )
        experts[name] = pipeline

    return experts
