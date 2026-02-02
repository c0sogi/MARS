import numpy as np
from library.pipeline_factory import (
    build_global_pipeline,
    build_global_rotational_pipeline,
    build_stratified_rotational_pipeline,
    build_factorized_interaction_pipeline,
    build_morphometric_pipeline,
)
from library.data_manager import get_feature_groups


class Expert:
    """
    A container for a candidate model configuration.
    """

    def __init__(self, name, pipeline, description=""):
        self.name = name
        self.pipeline = pipeline
        self.description = description

    def __repr__(self):
        return f"Expert(name='{self.name}')"


def generate_candidate_experts(feature_names):
    """
    Generates a library of candidate experts based on the SR-FIPE strategy.

    Args:
        feature_names (list): List of column names from the training data.

    Returns:
        list[Expert]: A list of Expert objects ready for training.
    """
    experts = []

    # 1. Identify Feature Groups
    groups = get_feature_groups(feature_names)

    # Define indices for Global views (excluding morphometrics)
    global_indices = groups["margin"] + groups["shape"] + groups["texture"]
    global_indices.sort()  # Ensure sorted order for consistency

    morph_indices = groups["morphometrics"]

    # Define Shrinkage Hyperparameters
    # 'auto' corresponds to Ledoit-Wolf in sklearn's lsqr solver
    shrinkage_levels = [0.001, 0.01, 0.1, "auto"]

    # =========================================================================
    # Group A: Global Statistical Anchors (The Baseline)
    # =========================================================================
    # 1. Marginal Topology (Power -> LDA)
    for s in shrinkage_levels:
        name = f"GroupA_Marginal_Shrinkage_{s}"
        pipeline = build_global_pipeline(global_indices, shrinkage=s)
        experts.append(
            Expert(name, pipeline, "Global Marginal View with PowerTransformer")
        )

    # 2. Global Rotational Topology (Power -> PCA -> Power -> LDA)
    for s in shrinkage_levels:
        name = f"GroupA_GlobalRotational_Shrinkage_{s}"
        pipeline = build_global_rotational_pipeline(global_indices, shrinkage=s)
        experts.append(Expert(name, pipeline, "Global Rotational View (PCA aligned)"))

    # =========================================================================
    # Group B: Stratified Rotational Experts (The Refinement)
    # =========================================================================
    # Independent Rotation of Margin, Shape, Texture
    for s in shrinkage_levels:
        name = f"GroupB_StratifiedRotational_Shrinkage_{s}"
        pipeline = build_stratified_rotational_pipeline(groups, shrinkage=s)
        experts.append(
            Expert(name, pipeline, "Stratified Rotational View (Independent PCA)")
        )

    # =========================================================================
    # Group C: Factorized Discriminative-Interaction Experts (The Innovation)
    # =========================================================================
    # Discriminative Bottleneck (LDA) -> Interactions -> LDA
    # We stick to n_components=10 as per design
    n_components = 10
    for s in shrinkage_levels:
        name = f"GroupC_FactorizedInteraction_N{n_components}_Shrinkage_{s}"
        pipeline = build_factorized_interaction_pipeline(
            groups, shrinkage=s, n_lda_components=n_components
        )
        experts.append(
            Expert(name, pipeline, "Factorized Interactions (Margin x Shape x Texture)")
        )

    # =========================================================================
    # Group D: Physical Polynomial Experts (The Domain Signal)
    # =========================================================================
    # Morphometrics -> Poly(2) -> LDA
    if morph_indices:
        for s in shrinkage_levels:
            name = f"GroupD_PhysicalPoly_Shrinkage_{s}"
            pipeline = build_morphometric_pipeline(morph_indices, shrinkage=s)
            experts.append(
                Expert(
                    name, pipeline, "Physical Morphometrics with Polynomial Features"
                )
            )

    return experts
