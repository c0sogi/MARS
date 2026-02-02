import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegressionCV
from sklearn.covariance import OAS
from library.config import FLOAT_PRECISION, RANDOM_SEED

# =============================================================================
# Topology Builders
# =============================================================================


def build_topology_A(estimator):
    """
    Topology A: Marginal Statistical Anchors.
    Input: Global View.
    Pipeline: PowerTransformer -> Estimator.
    """
    return Pipeline(
        [("pt", PowerTransformer(method="yeo-johnson")), ("clf", estimator)]
    )


def build_topology_B(estimator):
    """
    Topology B: Rotational Statistical Experts.
    Input: Global View.
    Pipeline: PowerTransformer -> PCA (Rotation) -> PowerTransformer -> Estimator.
    Note: PCA with whiten=False and all components acts as a rotation to principal axes.
    """
    return Pipeline(
        [
            ("pt1", PowerTransformer(method="yeo-johnson")),
            ("pca", PCA(n_components=None, whiten=False, random_state=RANDOM_SEED)),
            ("pt2", PowerTransformer(method="yeo-johnson")),
            ("clf", estimator),
        ]
    )


def build_topology_C(estimator, n_pca=25):
    """
    Topology C: Variance-Interaction Experts.
    Input: Global View.
    Pipeline: PowerTransformer -> PCA (Reduce) -> Poly (Degree 2) -> PowerTransformer -> Estimator.
    """
    return Pipeline(
        [
            ("pt1", PowerTransformer(method="yeo-johnson")),
            ("pca", PCA(n_components=n_pca, whiten=False, random_state=RANDOM_SEED)),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("pt2", PowerTransformer(method="yeo-johnson")),
            ("clf", estimator),
        ]
    )


def build_topology_D(estimator, n_lda=25):
    """
    Topology D: Separation-Interaction Experts.
    Input: Global View.
    Pipeline: PowerTransformer -> LDA (Reduce) -> Poly (Degree 2) -> PowerTransformer -> Estimator.
    """
    return Pipeline(
        [
            ("pt1", PowerTransformer(method="yeo-johnson")),
            ("lda_reduce", LinearDiscriminantAnalysis(n_components=n_lda)),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("pt2", PowerTransformer(method="yeo-johnson")),
            ("clf", estimator),
        ]
    )


def build_topology_E(estimator):
    """
    Topology E: Polynomial Physical Experts.
    Input: Morphometric View.
    Pipeline: PowerTransformer -> Poly (Degree 2) -> PowerTransformer -> Estimator.
    """
    return Pipeline(
        [
            ("pt1", PowerTransformer(method="yeo-johnson")),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("pt2", PowerTransformer(method="yeo-johnson")),
            ("clf", estimator),
        ]
    )


# =============================================================================
# Estimator Factories
# =============================================================================


def get_lda_estimator(strategy, value=None):
    """
    Returns an LDA estimator configured with a specific shrinkage strategy.
    Strategies: 'oas', 'auto' (Ledoit-Wolf), 'fixed'.
    """
    if strategy == "oas":
        # LDA with OAS covariance estimator
        return LinearDiscriminantAnalysis(solver="lsqr", covariance_estimator=OAS())
    elif strategy == "auto":
        # LDA with Ledoit-Wolf shrinkage
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    elif strategy == "fixed":
        # LDA with fixed shrinkage value
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage=value)
    else:
        raise ValueError(f"Unknown LDA strategy: {strategy}")


def get_logreg_estimator():
    """
    Returns a Logistic Regression estimator with built-in Cross-Validation.
    """
    return LogisticRegressionCV(
        Cs=10,
        cv=3,
        solver="lbfgs",
        max_iter=2000,
        scoring="neg_log_loss",
        multi_class="multinomial",
        n_jobs=2,  # Moderate parallelism within the expert
        random_state=RANDOM_SEED,
    )


# =============================================================================
# Expert Library Generation
# =============================================================================


def get_expert_library():
    """
    Generates the full library of expert configurations.

    Returns:
        list: A list of dictionaries, where each dictionary contains:
              - 'id': Unique identifier string.
              - 'view': 'global_view' or 'morph_view'.
              - 'pipeline': Unfitted sklearn Pipeline object.
    """
    library = []

    # Define Shrinkage Values for Fixed LDA
    shrinkage_values = [0.001, 0.01, 0.1, 0.5]

    # --- Topology A: Marginal Statistical Anchors (Global) ---
    # 1. LDA OAS
    library.append(
        {
            "id": "Topo_A_LDA_OAS",
            "view": "global_view",
            "pipeline": build_topology_A(get_lda_estimator("oas")),
        }
    )
    # 2. LDA Ledoit-Wolf
    library.append(
        {
            "id": "Topo_A_LDA_LW",
            "view": "global_view",
            "pipeline": build_topology_A(get_lda_estimator("auto")),
        }
    )
    # 3. LDA Fixed Shrinkage
    for s in shrinkage_values:
        library.append(
            {
                "id": f"Topo_A_LDA_Fixed_{s}",
                "view": "global_view",
                "pipeline": build_topology_A(get_lda_estimator("fixed", s)),
            }
        )

    # --- Topology B: Rotational Statistical Experts (Global) ---
    # 1. LDA OAS
    library.append(
        {
            "id": "Topo_B_LDA_OAS",
            "view": "global_view",
            "pipeline": build_topology_B(get_lda_estimator("oas")),
        }
    )
    # 2. LDA Fixed Shrinkage (Subset for diversity)
    for s in [0.01, 0.1]:
        library.append(
            {
                "id": f"Topo_B_LDA_Fixed_{s}",
                "view": "global_view",
                "pipeline": build_topology_B(get_lda_estimator("fixed", s)),
            }
        )

    # --- Topology C: Variance-Interaction Experts (Global) ---
    # 1. Logistic Regression CV
    library.append(
        {
            "id": "Topo_C_LogReg",
            "view": "global_view",
            "pipeline": build_topology_C(get_logreg_estimator()),
        }
    )
    # 2. LDA OAS
    library.append(
        {
            "id": "Topo_C_LDA_OAS",
            "view": "global_view",
            "pipeline": build_topology_C(get_lda_estimator("oas")),
        }
    )
    # 3. LDA Fixed Shrinkage (Low shrinkage often better for projected data)
    library.append(
        {
            "id": "Topo_C_LDA_Fixed_0.01",
            "view": "global_view",
            "pipeline": build_topology_C(get_lda_estimator("fixed", 0.01)),
        }
    )

    # --- Topology D: Separation-Interaction Experts (Global) ---
    # 1. Logistic Regression CV
    library.append(
        {
            "id": "Topo_D_LogReg",
            "view": "global_view",
            "pipeline": build_topology_D(get_logreg_estimator()),
        }
    )
    # 2. LDA OAS
    library.append(
        {
            "id": "Topo_D_LDA_OAS",
            "view": "global_view",
            "pipeline": build_topology_D(get_lda_estimator("oas")),
        }
    )

    # --- Topology E: Polynomial Physical Experts (Morphometric) ---
    # 1. LDA Ledoit-Wolf (Robust baseline for physics features)
    library.append(
        {
            "id": "Topo_E_LDA_LW",
            "view": "morph_view",
            "pipeline": build_topology_E(get_lda_estimator("auto")),
        }
    )
    # 2. LDA OAS
    library.append(
        {
            "id": "Topo_E_LDA_OAS",
            "view": "morph_view",
            "pipeline": build_topology_E(get_lda_estimator("oas")),
        }
    )
    # 3. Logistic Regression CV
    library.append(
        {
            "id": "Topo_E_LogReg",
            "view": "morph_view",
            "pipeline": build_topology_E(get_logreg_estimator()),
        }
    )

    return library
