import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer, PolynomialFeatures
from sklearn.decomposition import PCA
from library import config, transformers

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _instantiate_step(step_name, step_params):
    """
    Instantiates a scikit-learn transformer based on the config name and parameters.
    """
    # Clean step name (remove suffixes like _2 used for uniqueness in config)
    base_name = step_name.split("_")[0] if "_" in step_name else step_name

    if base_name == "power":
        return PowerTransformer(**step_params)
    elif base_name == "pca":
        return PCA(**step_params)
    elif base_name == "lda_transform":
        return transformers.LDADimensionalityReducer(**step_params)
    elif base_name == "poly":
        return PolynomialFeatures(**step_params)
    else:
        raise ValueError(f"Unknown step name: {step_name} (base: {base_name})")


def _build_linear_pipeline(steps_config):
    """
    Constructs a linear Pipeline from a list of (name, params) tuples.
    """
    steps = []
    # Always ensure float64 precision within the sub-pipeline
    steps.append(("float64_prec", transformers.Float64Transformer()))

    for step_name, step_params in steps_config:
        steps.append((step_name, _instantiate_step(step_name, step_params)))

    return Pipeline(steps)


# =============================================================================
# PIPELINE BUILDERS
# =============================================================================


def build_global_pipeline(pipeline_name, feature_subsets):
    """
    Builds a Group A pipeline (Global Statistical Anchors).

    Args:
        pipeline_name (str): Name of the specific pipeline variant (e.g., 'Marginal', 'Rotational').
        feature_subsets (dict): Dictionary mapping view names to column lists.

    Returns:
        sklearn.pipeline.Pipeline
    """
    # Find configuration
    pipeline_config = next(
        (p for p in config.GROUP_A_CONFIG["pipelines"] if p["name"] == pipeline_name),
        None,
    )
    if pipeline_config is None:
        raise ValueError(f"Global pipeline '{pipeline_name}' not found in config.")

    # 1. Select Global Features
    global_cols = feature_subsets["global"]

    # 2. Build Processing Steps
    processing_pipeline = _build_linear_pipeline(pipeline_config["steps"])

    # 3. Combine: Select -> Process
    # We use ColumnTransformer to select columns from the full X
    selector = ColumnTransformer(
        transformers=[("global_selector", processing_pipeline, global_cols)],
        remainder="drop",
    )

    # Wrap in a main pipeline
    return Pipeline(
        [
            ("root_float64", transformers.Float64Transformer()),
            ("selector_and_process", selector),
        ]
    )


def build_stratified_rotational_pipeline(feature_subsets):
    """
    Builds the Group B pipeline (Stratified Rotational Experts).
    Splits Margin, Shape, Texture; rotates them independently; concatenates.
    """
    conf = config.GROUP_B_CONFIG
    subsets = conf["subsets"]
    subset_pipeline_config = conf["subset_pipeline"]

    transformers_list = []
    for subset_name in subsets:
        cols = feature_subsets[subset_name]
        # Create independent rotation pipeline for this subset
        sub_pipe = _build_linear_pipeline(subset_pipeline_config)
        transformers_list.append((f"{subset_name}_rot", sub_pipe, cols))

    # ColumnTransformer handles the split-apply-combine logic
    stratified_processor = ColumnTransformer(
        transformers=transformers_list, remainder="drop"
    )

    return Pipeline(
        [
            ("root_float64", transformers.Float64Transformer()),
            ("stratified_rotational", stratified_processor),
        ]
    )


def build_intra_domain_pipeline(feature_subsets):
    """
    Builds the Group C Level 1 pipeline (Intra-Domain Interactions).
    Splits subsets; applies LDA projection + Poly expansion per subset; concatenates.
    """
    conf = config.GROUP_C_INTRA_CONFIG
    subsets = conf["subsets"]
    subset_pipeline_config = conf["subset_pipeline"]

    transformers_list = []
    for subset_name in subsets:
        cols = feature_subsets[subset_name]
        sub_pipe = _build_linear_pipeline(subset_pipeline_config)
        transformers_list.append((f"{subset_name}_intra", sub_pipe, cols))

    intra_processor = ColumnTransformer(
        transformers=transformers_list, remainder="drop"
    )

    return Pipeline(
        [
            ("root_float64", transformers.Float64Transformer()),
            ("intra_domain", intra_processor),
        ]
    )


def build_inter_domain_pipeline(pair, feature_subsets):
    """
    Builds a Group C Level 2 pipeline (Inter-Domain Interactions).

    Args:
        pair (tuple): Tuple of two subset names (e.g., ('margin', 'shape')).
        feature_subsets (dict): Feature mappings.
    """
    conf = config.GROUP_C_INTER_CONFIG
    subset_a, subset_b = pair

    # 1. Pre-Concatenation: Project each subset via LDA independently
    pre_concat_steps = conf["pre_concat_pipeline"]

    transformers_list = []

    # Subset A
    pipe_a = _build_linear_pipeline(pre_concat_steps)
    transformers_list.append((f"{subset_a}_pre", pipe_a, feature_subsets[subset_a]))

    # Subset B
    pipe_b = _build_linear_pipeline(pre_concat_steps)
    transformers_list.append((f"{subset_b}_pre", pipe_b, feature_subsets[subset_b]))

    pre_processor = ColumnTransformer(transformers=transformers_list, remainder="drop")

    # 2. Post-Concatenation: Interaction Polynomials + Normalization
    post_concat_pipe = _build_linear_pipeline(conf["post_concat_pipeline"])

    return Pipeline(
        [
            ("root_float64", transformers.Float64Transformer()),
            ("pre_process_pairs", pre_processor),
            ("post_process_interaction", post_concat_pipe),
        ]
    )


def build_morphometric_pipeline(feature_subsets):
    """
    Builds the Group D pipeline (Physical Polynomial Experts).
    Operates on extracted morphometrics.
    """
    conf = config.GROUP_D_CONFIG
    cols = feature_subsets["morphometrics"]

    # If no morphometrics exist (e.g. extraction failed), this might be empty.
    # However, data_loader ensures columns exist.

    processing_pipeline = _build_linear_pipeline(conf["pipeline"])

    selector = ColumnTransformer(
        transformers=[("morph_selector", processing_pipeline, cols)], remainder="drop"
    )

    return Pipeline(
        [
            ("root_float64", transformers.Float64Transformer()),
            ("morphometric_expert", selector),
        ]
    )
