from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.covariance import OAS
from library.config import SHRINKAGE_LIST, REG_PARAM_LIST


def create_expert_library():
    """
    Generates a library of expert models with varying inductive biases and feature views.

    The library consists of:
    1. Group A (Global View): LDA with various shrinkage estimators.
       - Robust baseline on high-dimensional data.
    2. Group B (Morph View): LDA and Regularized QDA.
       - Exploits low-dimensional probabilistic features.
       - QDA is allowed here due to favorable N/P ratio.
    3. Group C (Combined View): LDA with various shrinkage estimators.
       - Captures cross-covariance between shape and texture.

    Returns:
        list: A list of tuples, where each tuple is (Model Name, Classifier Instance, View Name).
    """
    experts = []

    def _get_lda_instance(shrinkage_param):
        """
        Helper to instantiate LDA with the correct solver and covariance estimator.
        """
        if shrinkage_param == "oas":
            # Use Oracle Approximating Shrinkage estimator
            # solver='lsqr' is required when using a custom covariance estimator
            return LinearDiscriminantAnalysis(solver="lsqr", covariance_estimator=OAS())
        else:
            # shrinkage_param is either 'auto' (Ledoit-Wolf) or a float (Fixed)
            # solver='lsqr' is required for shrinkage
            return LinearDiscriminantAnalysis(solver="lsqr", shrinkage=shrinkage_param)

    # -------------------------------------------------------------------------
    # Group A: The Global Linear Anchors
    # View: Global (192 features)
    # -------------------------------------------------------------------------
    for s in SHRINKAGE_LIST:
        model_name = f"LDA_Global_{s}"
        model = _get_lda_instance(s)
        experts.append((model_name, model, "Global"))

    # -------------------------------------------------------------------------
    # Group B: The Probabilistic Morphological Experts
    # View: Morph (22 features: Mean + Std of descriptors)
    # -------------------------------------------------------------------------
    # Subgroup B1: LDA on Morph
    for s in SHRINKAGE_LIST:
        model_name = f"LDA_Morph_{s}"
        model = _get_lda_instance(s)
        experts.append((model_name, model, "Morph"))

    # Subgroup B2: QDA on Morph (Regularized)
    # QDA is feasible here because dimensionality is low (~20)
    for r in REG_PARAM_LIST:
        model_name = f"QDA_Morph_reg{r}"
        # reg_param regularizes the covariance estimate per class
        model = QuadraticDiscriminantAnalysis(reg_param=r)
        experts.append((model_name, model, "Morph"))

    # -------------------------------------------------------------------------
    # Group C: The Synergistic Experts
    # View: Combined (Global + Morph)
    # -------------------------------------------------------------------------
    for s in SHRINKAGE_LIST:
        model_name = f"LDA_Combined_{s}"
        model = _get_lda_instance(s)
        experts.append((model_name, model, "Combined"))

    return experts
