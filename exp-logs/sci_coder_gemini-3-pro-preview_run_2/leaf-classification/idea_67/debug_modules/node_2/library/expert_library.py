import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    PowerTransformer,
    QuantileTransformer,
    PolynomialFeatures,
)
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from library.config import (
    LDA_SHRINKAGE_FIXED,
    LDA_SHRINKAGE_LIBRARY,
    QUANTILE_N_QUANTILES,
    QUANTILE_OUTPUT_DIST,
    LDA_N_COMPONENTS,
)
from library.custom_transformers import (
    Float64Transformer,
    StratifiedLDAReducer,
    GroupedInteractionTransformer,
)


def get_expert_library(feature_slices, priors=None):
    """
    Constructs the library of probabilistic experts based on the SM-FIPE strategy.

    Args:
        feature_slices (dict): Dictionary mapping feature group names to slice objects.
                               Expected keys: 'margin', 'shape', 'texture', 'morphometrics'.
        priors (np.ndarray, optional): Class priors to be used in LDA.

    Returns:
        list: A list of dictionaries, where each dictionary represents an expert:
              {
                  'name': str,
                  'pipeline': sklearn.pipeline.Pipeline,
                  'description': str,
                  'group': str
              }
    """
    experts = []

    # =========================================================================
    # Helper: Feature Selection
    # =========================================================================
    # Define indices for the "Global View" (Margin + Shape + Texture)
    # We assume the input X has columns in the order: Margin, Shape, Texture, Morphometrics
    # as defined in data_loader.py.

    # Extract slices
    sl_margin = feature_slices["margin"]
    sl_shape = feature_slices["shape"]
    sl_texture = feature_slices["texture"]
    sl_morph = feature_slices["morphometrics"]

    # Helper to create a ColumnTransformer that selects specific slices and applies a transformer
    def make_selector(transformer, slices_list):
        # ColumnTransformer requires a list of (name, transformer, columns) tuples.
        # We can pass slice objects directly as columns.
        transformers = []
        for i, sl in enumerate(slices_list):
            transformers.append((f"select_{i}", transformer, sl))
        return ColumnTransformer(transformers)

    # =========================================================================
    # Group A: Global Statistical Anchors
    # Input: Global View (Margin, Shape, Texture) -> 192 features
    # =========================================================================

    # We treat Margin, Shape, and Texture as one contiguous block for Group A
    # Note: ColumnTransformer concatenates results.
    global_slices = [sl_margin, sl_shape, sl_texture]

    # Topology A1: Marginal (PowerTransformer)
    # Topology A2: Global Rotational (Power -> PCA -> Power)
    # Topology A3: Robust (QuantileTransformer)

    # Define base feature processors

    # A1 Processor
    proc_a1 = Pipeline([("power", PowerTransformer(method="yeo-johnson"))])

    # A2 Processor
    proc_a2 = Pipeline(
        [
            ("power_in", PowerTransformer(method="yeo-johnson")),
            ("pca", PCA(whiten=False, random_state=42)),
            ("power_out", PowerTransformer(method="yeo-johnson")),
        ]
    )

    # A3 Processor
    proc_a3 = Pipeline(
        [
            (
                "quantile",
                QuantileTransformer(
                    n_quantiles=QUANTILE_N_QUANTILES,
                    output_distribution=QUANTILE_OUTPUT_DIST,
                    random_state=42,
                ),
            )
        ]
    )

    # Generate Experts for Group A (Fixed Shrinkage)
    topologies_a = [
        ("A1_Marginal", proc_a1),
        ("A2_GlobalRot", proc_a2),
        ("A3_Robust", proc_a3),
    ]

    for topo_name, proc in topologies_a:
        for shrinkage in LDA_SHRINKAGE_FIXED:
            name = f"{topo_name}_shrink_{shrinkage}"

            # Construct Pipeline
            # 1. Float64 Cast
            # 2. Select Global Features & Apply Processor
            # 3. LDA Classifier

            # Note: We use ColumnTransformer to select the 3 slices and apply the processor to them jointly.
            # However, ColumnTransformer applies to each column subset independently if listed separately.
            # To apply to the UNION of slices globally, we need to pass the list of all indices or use a single slice if contiguous.
            # Since data_loader constructs them contiguously (Margin->Shape->Texture), we can create a unified slice.
            # But to be safe and generic, we'll use a wrapper that selects then transforms.

            # Actually, simplest way: ColumnTransformer with one entry selecting all relevant columns.
            # We need indices for that.
            global_indices = []
            for sl in global_slices:
                global_indices.extend(range(sl.start, sl.stop))

            pipeline = Pipeline(
                [
                    ("float64", Float64Transformer()),
                    (
                        "preproc",
                        ColumnTransformer([("global_proc", proc, global_indices)]),
                    ),
                    (
                        "clf",
                        LinearDiscriminantAnalysis(
                            solver="lsqr", shrinkage=shrinkage, priors=priors
                        ),
                    ),
                ]
            )

            experts.append(
                {
                    "name": name,
                    "pipeline": pipeline,
                    "description": f"Group A: {topo_name} with shrinkage {shrinkage}",
                    "group": "A",
                }
            )

    # =========================================================================
    # Group B: Stratified Rotational Experts
    # Input: Global View, but processed independently per group
    # =========================================================================

    # Rotational Pipeline Definition
    def make_rotational_pipeline():
        return Pipeline(
            [
                ("power_in", PowerTransformer(method="yeo-johnson")),
                ("pca", PCA(whiten=False, random_state=42)),
                ("power_out", PowerTransformer(method="yeo-johnson")),
            ]
        )

    # Stratified Preprocessor
    stratified_preproc = ColumnTransformer(
        [
            ("margin_rot", make_rotational_pipeline(), sl_margin),
            ("shape_rot", make_rotational_pipeline(), sl_shape),
            ("texture_rot", make_rotational_pipeline(), sl_texture),
        ]
    )

    for shrinkage in LDA_SHRINKAGE_LIBRARY:
        name = f"B_StratifiedRot_shrink_{shrinkage}"

        pipeline = Pipeline(
            [
                ("float64", Float64Transformer()),
                ("stratified_proc", stratified_preproc),
                (
                    "clf",
                    LinearDiscriminantAnalysis(
                        solver="lsqr", shrinkage=shrinkage, priors=priors
                    ),
                ),
            ]
        )

        experts.append(
            {
                "name": name,
                "pipeline": pipeline,
                "description": f"Group B: Stratified Rotational with shrinkage {shrinkage}",
                "group": "B",
            }
        )

    # =========================================================================
    # Group C: Factorized Discriminative-Interaction Experts
    # Input: Global View
    # Pipeline: StratifiedLDA -> Interactions -> Power -> LDA
    # =========================================================================

    # 1. Stratified LDA Reducer (Bottleneck)
    # This reduces each group (Margin, Shape, Texture) to LDA_N_COMPONENTS (9) features.
    # The output will be a concatenation of these 3 reduced vectors.
    # Output shape: (N_samples, 27) where 0-9 is Margin, 9-18 is Shape, 18-27 is Texture.

    stratified_reducer = StratifiedLDAReducer(
        feature_slices={"margin": sl_margin, "shape": sl_shape, "texture": sl_texture},
        n_components=LDA_N_COMPONENTS,
        solver="svd",  # SVD is standard for LDA transformation
        priors=priors,
    )

    # 2. Interaction Transformer
    # We need to define slices for the *intermediate* representation output by stratified_reducer.
    n_comp = LDA_N_COMPONENTS
    intermediate_slices = {
        "margin": slice(0, n_comp),
        "shape": slice(n_comp, n_comp * 2),
        "texture": slice(n_comp * 2, n_comp * 3),
    }

    interaction_pairs = [
        ("margin", "texture"),
        ("shape", "texture"),
        ("margin", "shape"),
    ]

    interaction_transformer = GroupedInteractionTransformer(
        feature_slices=intermediate_slices, interaction_pairs=interaction_pairs
    )

    # 3. Full Pipeline
    # Note: We apply PowerTransformer after interactions to Gaussianize the product features.

    # We also include the original reduced features? The prompt says "Interaction Synthesis... Re-Gaussianization".
    # Usually, interactions are added to main effects. However, the description implies a dense optimized subspace.
    # "Construct a library... Factorized Discriminative-Interaction Experts".
    # Given the complexity, let's stick to the interactions as the primary features for this expert group,
    # or concatenate them. The prompt says "Compute pairwise cross-domain interactions... Re-Gaussianization".
    # It doesn't explicitly say "concatenate with original". I will use interactions only to be specific to this expert type,
    # as Group B already covers the main effects (rotational).

    for shrinkage in LDA_SHRINKAGE_LIBRARY:
        name = f"C_FactorizedInteraction_shrink_{shrinkage}"

        pipeline = Pipeline(
            [
                ("float64", Float64Transformer()),
                ("bottleneck", stratified_reducer),
                ("interactions", interaction_transformer),
                ("power", PowerTransformer(method="yeo-johnson")),
                (
                    "clf",
                    LinearDiscriminantAnalysis(
                        solver="lsqr", shrinkage=shrinkage, priors=priors
                    ),
                ),
            ]
        )

        experts.append(
            {
                "name": name,
                "pipeline": pipeline,
                "description": f"Group C: Factorized Interactions with shrinkage {shrinkage}",
                "group": "C",
            }
        )

    # =========================================================================
    # Group D: Physical Polynomial Experts
    # Input: Morphometrics Only
    # =========================================================================

    # Pipeline: Power -> Poly(2) -> Power
    morph_proc = Pipeline(
        [
            ("power_in", PowerTransformer(method="yeo-johnson")),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("power_out", PowerTransformer(method="yeo-johnson")),
        ]
    )

    # We use 'auto' (Ledoit-Wolf) for this group as suggested in description
    # "Algorithm: LDA (Ledoit-Wolf)"

    name = "D_PhysicalPoly_LedoitWolf"

    pipeline = Pipeline(
        [
            ("float64", Float64Transformer()),
            ("select_morph", ColumnTransformer([("morph_proc", morph_proc, sl_morph)])),
            (
                "clf",
                LinearDiscriminantAnalysis(
                    solver="lsqr", shrinkage="auto", priors=priors
                ),
            ),
        ]
    )

    experts.append(
        {
            "name": name,
            "pipeline": pipeline,
            "description": "Group D: Physical Polynomial with Ledoit-Wolf shrinkage",
            "group": "D",
        }
    )

    return experts
