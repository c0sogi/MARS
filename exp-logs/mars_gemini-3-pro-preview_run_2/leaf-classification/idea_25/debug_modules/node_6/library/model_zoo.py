import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.covariance import OAS
from library.config import LDA_CONFIGS, QDA_CONFIGS, GNB_CONFIGS


class LDA_OAS_Wrapper(BaseEstimator, ClassifierMixin):
    """
    A wrapper for LinearDiscriminantAnalysis that uses the Oracle Approximating Shrinkage (OAS)
    estimator to determine the optimal shrinkage coefficient before fitting.
    """

    def __init__(self, solver="lsqr"):
        self.solver = solver
        self.shrinkage_ = None
        self.lda_ = None
        self.classes_ = None

    def fit(self, X, y):
        """
        Estimates shrinkage using OAS on X, then fits LDA.
        """
        # Estimate shrinkage coefficient using OAS
        # OAS computes the shrinkage for the global covariance matrix.
        # We use this as a robust heuristic for the LDA within-class shrinkage.
        oas = OAS()
        oas.fit(X)
        self.shrinkage_ = oas.shrinkage_

        # Initialize and fit standard LDA with the computed shrinkage
        self.lda_ = LinearDiscriminantAnalysis(
            solver=self.solver, shrinkage=self.shrinkage_
        )
        self.lda_.fit(X, y)

        # Expose classes_ attribute for compatibility with sklearn utilities
        self.classes_ = self.lda_.classes_
        return self

    def predict(self, X):
        """Delegates prediction to the underlying LDA model."""
        if self.lda_ is None:
            raise RuntimeError("Model not fitted yet.")
        return self.lda_.predict(X)

    def predict_proba(self, X):
        """Delegates probability prediction to the underlying LDA model."""
        if self.lda_ is None:
            raise RuntimeError("Model not fitted yet.")
        return self.lda_.predict_proba(X)


def get_expert_library():
    """
    Constructs and returns the library of probabilistic expert models.

    Iterates through configurations for LDA, QDA, and GNB defined in library.config.
    Creates an instance of each model for every feature view ('Global', 'Morph', 'Combined').

    Returns:
        list[dict]: A list of expert definitions. Each dict contains:
            - 'name': Unique string identifier.
            - 'model': The instantiated scikit-learn compatible model.
            - 'view': The feature view the model should be trained on.
    """
    experts = []
    views = ["Global", "Morph", "Combined"]

    # -------------------------------------------------------------------------
    # Group A: Linear Discriminant Analysis (The Linear Anchors)
    # -------------------------------------------------------------------------
    for view in views:
        for config in LDA_CONFIGS:
            solver = config.get("solver", "lsqr")
            shrinkage = config.get("shrinkage", "auto")

            # Handle special OAS case
            if shrinkage == "OAS":
                model = LDA_OAS_Wrapper(solver=solver)
                name = f"LDA_{view}_OAS"
            else:
                # Standard LDA with Fixed or Auto (Ledoit-Wolf) shrinkage
                model = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)
                # Format name based on shrinkage type
                if shrinkage == "auto":
                    shrink_str = "auto"
                else:
                    shrink_str = f"{float(shrinkage):.3f}"
                name = f"LDA_{view}_shrink_{shrink_str}"

            experts.append({"name": name, "model": model, "view": view})

    # -------------------------------------------------------------------------
    # Group B: Quadratic Discriminant Analysis (The Quadratic Innovators)
    # -------------------------------------------------------------------------
    for view in views:
        for config in QDA_CONFIGS:
            reg_param = config.get("reg_param", 0.0)

            model = QuadraticDiscriminantAnalysis(reg_param=reg_param)
            name = f"QDA_{view}_reg_{reg_param:.2f}"

            experts.append({"name": name, "model": model, "view": view})

    # -------------------------------------------------------------------------
    # Group C: Gaussian Naive Bayes (The Diagonal Stabilizers)
    # -------------------------------------------------------------------------
    for view in views:
        for config in GNB_CONFIGS:
            var_smoothing = config.get("var_smoothing", 1e-9)

            model = GaussianNB(var_smoothing=var_smoothing)
            name = f"GNB_{view}_smooth_{var_smoothing:.0e}"

            experts.append({"name": name, "model": model, "view": view})

    return experts
