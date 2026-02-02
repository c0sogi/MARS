import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from library.pipeline_factory import (
    get_global_pipeline,
    get_stratified_rotational_pipeline,
    get_poly_pipeline,
)
from library.config import GRID_GROUP_A, GRID_GROUP_B, GRID_GROUP_C


class LeafExpert(BaseEstimator, ClassifierMixin):
    """
    A wrapper class that encapsulates a specific feature subset, a preprocessing pipeline,
    and a classifier. This ensures that each expert operates on its designated view
    of the data with the correct transformation topology.
    """

    def __init__(self, name, pipeline, estimator, feature_indices):
        """
        Args:
            name (str): Unique identifier for the expert.
            pipeline (sklearn.pipeline.Pipeline): Preprocessing pipeline.
            estimator (sklearn.base.BaseEstimator): The classifier (LDA or QDA).
            feature_indices (list or np.ndarray): Indices of columns to use from the input matrix.
        """
        self.name = name
        self.pipeline = pipeline
        self.estimator = estimator
        self.feature_indices = feature_indices

    def fit(self, X, y):
        """
        Fits the pipeline and estimator on the specified feature subset.
        """
        # Select specific features for this expert
        X_subset = X[:, self.feature_indices]

        # Fit and transform using the pipeline
        X_transformed = self.pipeline.fit_transform(X_subset, y)

        # Fit the estimator
        self.estimator.fit(X_transformed, y)
        return self

    def predict(self, X):
        """
        Predicts class labels.
        """
        X_subset = X[:, self.feature_indices]
        X_transformed = self.pipeline.transform(X_subset)
        return self.estimator.predict(X_transformed)

    def predict_proba(self, X):
        """
        Predicts class probabilities.
        """
        X_subset = X[:, self.feature_indices]
        X_transformed = self.pipeline.transform(X_subset)
        return self.estimator.predict_proba(X_transformed)


def generate_expert_library(feature_groups):
    """
    Generates the library of experts based on the FR-SPPE strategy.

    Args:
        feature_groups (dict): Mapping of group names to column indices.

    Returns:
        list: A list of initialized LeafExpert objects.
    """
    experts = []

    # 1. Identify Indices
    # Original 192 features (Margin + Shape + Texture)
    # We combine these to form the 'Global' view for Group A.
    original_indices = sorted(
        feature_groups["margin"] + feature_groups["shape"] + feature_groups["texture"]
    )

    morph_indices = feature_groups["morphometrics"]

    # =========================================================================
    # Group A: Global Linear Anchors
    # Input: Global View (192 features)
    # Algorithms: LDA + OAS (Fixed Shrinkage)
    # =========================================================================

    # Pipeline 1: Marginal (Yeo-Johnson only)
    pipe_marginal = get_global_pipeline()

    # Pipeline 2: Stratified-Rotational (Split -> Rot -> Recombine)
    pipe_rotational = get_stratified_rotational_pipeline(feature_groups)

    for shrinkage in GRID_GROUP_A:
        # Expert A1: Marginal Topology
        lda_marginal = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=shrinkage)
        experts.append(
            LeafExpert(
                name=f"GroupA_Marginal_Shrink{shrinkage}",
                pipeline=clone(pipe_marginal),
                estimator=lda_marginal,
                feature_indices=original_indices,
            )
        )

        # Expert A2: Rotational Topology
        lda_rotational = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=shrinkage)
        experts.append(
            LeafExpert(
                name=f"GroupA_Rotational_Shrink{shrinkage}",
                pipeline=clone(pipe_rotational),
                estimator=lda_rotational,
                feature_indices=original_indices,
            )
        )

    # =========================================================================
    # Group B: Physical Polynomial Experts
    # Input: Polarity-Corrected Morphometrics
    # Algorithms: Regularized QDA
    # =========================================================================

    # Polynomial Expansion Degree 2
    pipe_poly_b = get_poly_pipeline(degree=2)

    for reg_param in GRID_GROUP_B:
        qda = QuadraticDiscriminantAnalysis(reg_param=reg_param)
        experts.append(
            LeafExpert(
                name=f"GroupB_Morph_Poly_Reg{reg_param}",
                pipeline=clone(pipe_poly_b),
                estimator=qda,
                feature_indices=morph_indices,
            )
        )

    # =========================================================================
    # Group C: Stratified Full-Rank Polynomial Experts
    # Input: Independent subsets (Margin, Shape, Texture)
    # Algorithms: LDA + OAS (High Shrinkage)
    # =========================================================================

    pipe_poly_c = get_poly_pipeline(degree=2)
    target_domains = ["margin", "shape", "texture"]

    for domain in target_domains:
        domain_indices = feature_groups[domain]
        if not domain_indices:
            continue

        for shrinkage in GRID_GROUP_C:
            lda_c = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=shrinkage)
            experts.append(
                LeafExpert(
                    name=f"GroupC_{domain.capitalize()}_Poly_Shrink{shrinkage}",
                    pipeline=clone(pipe_poly_c),
                    estimator=lda_c,
                    feature_indices=domain_indices,
                )
            )

    return experts
