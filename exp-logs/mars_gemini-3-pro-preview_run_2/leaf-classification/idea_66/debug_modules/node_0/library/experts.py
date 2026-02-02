from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    PowerTransformer,
    QuantileTransformer,
    PolynomialFeatures,
)
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from library.transformers import Float64Wrapper, FactorizedDiscriminantProjector
from library.config import RANDOM_SEED


def get_expert_library():
    """
    Constructs and returns the library of candidate experts for the ensemble.

    Each expert is defined by:
    - name: Unique identifier.
    - input_type: The source of input data ('provided_features' or 'morphometrics').
    - pipeline: The preprocessing steps (sklearn Pipeline).
    - estimator: The final classification model (sklearn Estimator).

    Returns:
        list[dict]: A list of expert configuration dictionaries.
    """
    experts = []

    # Define shrinkage parameters for LDA
    # 'auto' corresponds to the Ledoit-Wolf lemma for analytic shrinkage optimization.
    # Fixed values [0.001, 0.01] provide regularization anchors.
    shrinkage_options = ["auto", 0.001, 0.01]

    # =========================================================================
    # Group A: Global Statistical Anchors
    # Input: Provided Features (192 columns: Margin, Shape, Texture)
    # Goal: Preserve state-of-the-art baseline using global linear correlations.
    # =========================================================================

    # Topology 1: Marginal (PowerTransformer -> LDA)
    # Stabilizes variance feature-wise.
    for s in shrinkage_options:
        name = f"GroupA_Marginal_Shrinkage_{s}"
        pipeline = make_pipeline(
            Float64Wrapper(), PowerTransformer(method="yeo-johnson")
        )
        estimator = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=s)

        experts.append(
            {
                "name": name,
                "input_type": "provided_features",
                "pipeline": pipeline,
                "estimator": estimator,
            }
        )

    # Topology 2: Rotational (Power -> PCA -> Power -> LDA)
    # Aligns data with principal axes, approximating multivariate normality.
    # whiten=False avoids noise amplification.
    for s in shrinkage_options:
        name = f"GroupA_Rotational_Shrinkage_{s}"
        pipeline = make_pipeline(
            Float64Wrapper(),
            PowerTransformer(method="yeo-johnson"),
            PCA(whiten=False, random_state=RANDOM_SEED),
            PowerTransformer(method="yeo-johnson"),
        )
        estimator = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=s)

        experts.append(
            {
                "name": name,
                "input_type": "provided_features",
                "pipeline": pipeline,
                "estimator": estimator,
            }
        )

    # Topology 3: Robust (QuantileTransformer -> LDA)
    # Constrains rank-based normalization to handle skew/outliers without overfitting.
    for s in shrinkage_options:
        name = f"GroupA_Robust_Shrinkage_{s}"
        pipeline = make_pipeline(
            Float64Wrapper(),
            QuantileTransformer(
                output_distribution="normal", n_quantiles=50, random_state=RANDOM_SEED
            ),
        )
        estimator = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=s)

        experts.append(
            {
                "name": name,
                "input_type": "provided_features",
                "pipeline": pipeline,
                "estimator": estimator,
            }
        )

    # =========================================================================
    # Group B: Physical Polynomial Experts
    # Input: Morphometrics (Hu Moments + Geometric Scalars)
    # Goal: Capture non-linear physical constraints (e.g., Solidity * Eccentricity).
    # =========================================================================

    # Only using 'auto' shrinkage (Ledoit-Wolf) as this view is lower dimensional
    # and physically meaningful, making analytic shrinkage robust.
    name = "GroupB_Morphometric_Poly_Auto"
    pipeline = make_pipeline(
        Float64Wrapper(),
        PowerTransformer(method="yeo-johnson"),
        PolynomialFeatures(degree=2, include_bias=False),
        PowerTransformer(method="yeo-johnson"),
    )
    estimator = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")

    experts.append(
        {
            "name": name,
            "input_type": "morphometrics",
            "pipeline": pipeline,
            "estimator": estimator,
        }
    )

    # =========================================================================
    # Group C: Factorized-Discriminative Interaction Experts
    # Input: Provided Features (Split internally by FactorizedDiscriminantProjector)
    # Goal: Explicitly model quadratic dependencies between semantic groups
    #       (Margin, Shape, Texture) in a dense, optimized subspace.
    # =========================================================================

    for s in shrinkage_options:
        name = f"GroupC_Factorized_Interaction_Shrinkage_{s}"
        pipeline = make_pipeline(
            Float64Wrapper(),
            # Project semantic groups to 9 discriminative components each
            FactorizedDiscriminantProjector(n_components=9),
            # Expand interactions (approx 400 features)
            PolynomialFeatures(degree=2, include_bias=False),
            # Re-Gaussianize before final LDA
            PowerTransformer(method="yeo-johnson"),
        )
        estimator = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=s)

        experts.append(
            {
                "name": name,
                "input_type": "provided_features",
                "pipeline": pipeline,
                "estimator": estimator,
            }
        )

    return experts
