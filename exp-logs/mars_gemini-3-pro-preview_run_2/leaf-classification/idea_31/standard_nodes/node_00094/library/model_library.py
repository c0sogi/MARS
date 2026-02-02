import numpy as np
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.covariance import OAS
from library import config


class BaseExpert:
    """
    Base class for all experts in the SCPGE ensemble.
    Wraps an sklearn estimator and attaches metadata about the view it consumes.
    """

    def __init__(self, name, view_type, model):
        self.name = name
        self.view_type = view_type  # 'global', 'macro', or 'combined'
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def __repr__(self):
        return f"<{self.name}>"


class LDAExpert(BaseExpert):
    """
    Tier 1 Expert: Shared-Covariance Anchor.
    Uses Linear Discriminant Analysis with various shrinkage estimators.
    Supports 'auto' (Ledoit-Wolf), 'oas' (Oracle Approximating Shrinkage), and fixed float shrinkage.
    """

    def __init__(self, shrinkage, view_type="global"):
        name = f"LDA_{view_type}_shrinkage={shrinkage}"

        # Configure solver and shrinkage method
        # Note: sklearn 1.2+ supports covariance_estimator for 'lsqr' solver
        if shrinkage == "oas":
            # Use OAS for covariance estimation
            model = LinearDiscriminantAnalysis(
                solver=config.LDA_SOLVER, covariance_estimator=OAS()
            )
        else:
            # Use standard shrinkage ('auto' or float)
            model = LinearDiscriminantAnalysis(
                solver=config.LDA_SOLVER, shrinkage=shrinkage
            )

        super().__init__(name, view_type, model)


class PCQDAExpert(BaseExpert):
    """
    Tier 2 Expert: Global Class-Specific Expert.
    Projects high-dimensional global features into a dense Principal Subspace
    to enable stable QDA (Quadratic Discriminant Analysis).
    """

    def __init__(self, n_components, reg_param, view_type="global"):
        name = f"PCQDA_{view_type}_pca={n_components}_reg={reg_param}"

        # Pipeline: PCA -> QDA
        steps = [
            ("pca", PCA(n_components=n_components, random_state=config.RANDOM_STATE)),
            ("qda", QuadraticDiscriminantAnalysis(reg_param=reg_param)),
        ]
        model = Pipeline(steps)

        super().__init__(name, view_type, model)


class MacroQDAExpert(BaseExpert):
    """
    Tier 3 Expert: Macro Class-Specific Expert.
    Applies QDA directly to the low-dimensional Morphometric (Macro) features.
    """

    def __init__(self, reg_param, view_type="macro"):
        name = f"MacroQDA_{view_type}_reg={reg_param}"
        model = QuadraticDiscriminantAnalysis(reg_param=reg_param)
        super().__init__(name, view_type, model)


class GNBExpert(BaseExpert):
    """
    Tier 4 Expert: Diagonal Covariance Anchor.
    Uses Gaussian Naive Bayes as a high-bias failsafe.
    """

    def __init__(self, var_smoothing, view_type="global"):
        name = f"GNB_{view_type}_smooth={var_smoothing}"
        model = GaussianNB(var_smoothing=var_smoothing)
        super().__init__(name, view_type, model)


def get_expert_library():
    """
    Generates the complete library of probabilistic experts spanning the
    Covariance Complexity Spectrum as defined in config.py.

    Returns:
        list: A list of instantiated Expert objects (LDA, PC-QDA, Macro-QDA, GNB).
    """
    experts = []

    # -------------------------------------------------------------------------
    # Tier 1: Shared-Covariance Anchors (LDA)
    # Applied to both Global and Combined views
    # -------------------------------------------------------------------------
    for shrinkage in config.LDA_SHRINKAGE_GRID:
        experts.append(LDAExpert(shrinkage, view_type="global"))
        experts.append(LDAExpert(shrinkage, view_type="combined"))

    # -------------------------------------------------------------------------
    # Tier 2: Global Class-Specific Experts (PC-QDA)
    # Applied to Global view (dimensionality reduction required)
    # -------------------------------------------------------------------------
    for n_components in config.PCA_N_COMPONENTS_GRID:
        for reg_param in config.PCA_QDA_REG_PARAM_GRID:
            experts.append(PCQDAExpert(n_components, reg_param, view_type="global"))

    # -------------------------------------------------------------------------
    # Tier 3: Macro Class-Specific Experts (QDA)
    # Applied to Macro view (low dimensionality allows direct QDA)
    # -------------------------------------------------------------------------
    for reg_param in config.MACRO_QDA_REG_PARAM_GRID:
        experts.append(MacroQDAExpert(reg_param, view_type="macro"))

    # -------------------------------------------------------------------------
    # Tier 4: Diagonal Covariance Anchors (GNB)
    # Applied to Global view
    # -------------------------------------------------------------------------
    for var_smoothing in config.GNB_VAR_SMOOTHING_GRID:
        experts.append(GNBExpert(var_smoothing, view_type="global"))

    return experts
