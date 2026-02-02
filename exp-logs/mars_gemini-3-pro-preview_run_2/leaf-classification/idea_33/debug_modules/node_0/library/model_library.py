import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from library import config, data_loader


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Custom transformer to select specific feature columns from a pandas DataFrame.
    Ensures that experts only see the view of the data they are assigned to (Global vs Macro).
    """

    def __init__(self, feature_names):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Expecting X to be a pandas DataFrame as per data_loader output
        if isinstance(X, pd.DataFrame):
            # Check for missing columns to provide helpful error messages
            missing_cols = [col for col in self.feature_names if col not in X.columns]
            if missing_cols:
                raise ValueError(
                    f"FeatureSelector: The following required columns are missing: {missing_cols}"
                )
            return X[self.feature_names].values
        else:
            raise TypeError("FeatureSelector expects a pandas DataFrame input.")


def get_pipeline(feature_group_name, norm_strategy, shrinkage):
    """
    Constructs a scikit-learn pipeline for a specific expert configuration.

    Args:
        feature_group_name (str): 'global' or 'macro'.
        norm_strategy (str): 'parametric' (Yeo-Johnson) or 'non_parametric' (Quantile).
        shrinkage (str or float): LDA shrinkage parameter ('auto' for Ledoit-Wolf, or a float).

    Returns:
        sklearn.pipeline.Pipeline: The constructed pipeline.
    """
    # 1. Retrieve Feature Names
    feature_groups = data_loader.get_feature_groups()
    if feature_group_name not in feature_groups:
        raise ValueError(f"Unknown feature group: {feature_group_name}")

    selected_features = feature_groups[feature_group_name]

    # 2. Define Pipeline Steps
    steps = []

    # Step 1: Feature Selection
    steps.append(("selector", FeatureSelector(feature_names=selected_features)))

    # Step 2: Gaussianization Strategy
    if norm_strategy == "parametric":
        # Group A & C: PowerTransformer (Yeo-Johnson)
        # This forces data to be normal-like using power laws.
        steps.append(("scaler", PowerTransformer(method=config.POWER_METHOD)))

    elif norm_strategy == "non_parametric":
        # Group B: Regularized Quantile Transformer
        # n_quantiles is constrained (e.g., 50) to prevent overfitting to the empirical distribution
        # while handling multi-modal or highly skewed data that Yeo-Johnson misses.
        steps.append(
            (
                "scaler",
                QuantileTransformer(
                    n_quantiles=config.N_QUANTILES,
                    output_distribution=config.QUANTILE_OUTPUT_DIST,
                    random_state=config.RANDOM_STATE,
                ),
            )
        )
    else:
        raise ValueError(f"Unknown normalization strategy: {norm_strategy}")

    # Step 3: Covariance Expert (LDA)
    # solver='lsqr' supports shrinkage.
    # shrinkage='auto' implements the Ledoit-Wolf lemma.
    lda = LinearDiscriminantAnalysis(solver=config.LDA_SOLVER, shrinkage=shrinkage)
    steps.append(("clf", lda))

    return Pipeline(steps)


def build_expert_library():
    """
    Constructs the full library of experts for the Dual-Gaussianized Ensemble.

    Returns:
        list of dict: A list where each dictionary defines an expert's configuration.
    """
    experts = []

    # Combine automatic (Ledoit-Wolf) and fixed shrinkage candidates
    shrinkage_options = config.LDA_AUTOMATIC_SHRINKAGE + config.LDA_SHRINKAGE_CANDIDATES

    # ==========================================================================
    # Group A: Parametric Gaussian Anchors (The Baseline)
    # Features: Global (192)
    # Strategy: Yeo-Johnson -> LDA (Various Shrinkage)
    # ==========================================================================
    for shrinkage in shrinkage_options:
        # Create a readable name for the expert
        s_name = "LW" if shrinkage == "auto" else f"Fixed_{shrinkage}"
        expert_name = f"GroupA_Parametric_Global_{s_name}"

        pipeline = get_pipeline(
            feature_group_name="global", norm_strategy="parametric", shrinkage=shrinkage
        )

        experts.append(
            {
                "name": expert_name,
                "pipeline": pipeline,
                "feature_group": "global",
                "type": "parametric_anchor",
            }
        )

    # ==========================================================================
    # Group B: Regularized Non-Parametric Experts (The Innovation)
    # Features: Global (192)
    # Strategy: QuantileTransformer(n=50) -> LDA (Various Shrinkage)
    # ==========================================================================
    for shrinkage in shrinkage_options:
        s_name = "LW" if shrinkage == "auto" else f"Fixed_{shrinkage}"
        expert_name = f"GroupB_NonParametric_Global_{s_name}"

        pipeline = get_pipeline(
            feature_group_name="global",
            norm_strategy="non_parametric",
            shrinkage=shrinkage,
        )

        experts.append(
            {
                "name": expert_name,
                "pipeline": pipeline,
                "feature_group": "global",
                "type": "non_parametric_expert",
            }
        )

    # ==========================================================================
    # Group C: Orthogonal Morphometric Experts
    # Features: Macro (11)
    # Strategy: Yeo-Johnson -> LDA (Ledoit-Wolf)
    # ==========================================================================
    # We only use 'auto' (Ledoit-Wolf) here as the macro features are low-dimensional
    # and physically constrained, so complex shrinkage tuning is less critical.
    expert_name = "GroupC_Morphometric_Macro_LW"

    pipeline = get_pipeline(
        feature_group_name="macro", norm_strategy="parametric", shrinkage="auto"
    )

    experts.append(
        {
            "name": expert_name,
            "pipeline": pipeline,
            "feature_group": "macro",
            "type": "morphometric_expert",
        }
    )

    return experts
