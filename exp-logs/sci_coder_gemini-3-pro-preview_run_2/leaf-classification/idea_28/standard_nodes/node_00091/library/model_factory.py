import numpy as np
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from library.config import SHRINKAGE_GRID, RANDOM_SEED

# Grid for Quadratic Discriminant Analysis Regularization
# Used only for the low-dimensional Macro view
QDA_REG_GRID = [0.0, 0.01, 0.05, 0.1, 0.5]


class GenerativeExpert:
    """
    A wrapper for Generative Models (LDA/QDA) to ensure strict float64 precision
    and numerical stability for Log Loss optimization.
    """

    def __init__(self, model):
        """
        Args:
            model: An instantiated sklearn estimator (LDA or QDA).
        """
        self.model = model

    def fit(self, X, y):
        """
        Fit the underlying model.
        Args:
            X (np.ndarray): Feature matrix (will be cast to float64).
            y (np.ndarray): Target vector.
        """
        # Ensure input is float64 for maximum precision
        X_f64 = X.astype(np.float64)
        self.model.fit(X_f64, y)
        return self

    def predict_proba(self, X):
        """
        Predict class probabilities with numerical clipping.
        Args:
            X (np.ndarray): Feature matrix.
        Returns:
            np.ndarray: Probability matrix (float64), clipped to [1e-15, 1-1e-15].
        """
        X_f64 = X.astype(np.float64)
        probas = self.model.predict_proba(X_f64)

        # Clip probabilities to avoid log(0) in metric calculation
        # The task description specifies this clipping for the metric.
        # We apply it here to ensure the greedy selector sees valid log-loss values.
        probas = np.clip(probas, 1e-15, 1 - 1e-15)

        return probas.astype(np.float64)


def build_expert_library():
    """
    Constructs the library of probabilistic experts for the Multi-Resolution Ensemble.

    Returns:
        dict: A dictionary where keys are unique expert IDs and values are dictionaries
              containing the 'model' (GenerativeExpert) and the target 'view'.

              Format:
              {
                  "expert_id": {
                      "model": GenerativeExpert(...),
                      "view": "macro" | "micro" | "synergistic"
                  },
                  ...
              }
    """
    experts = {}

    # =========================================================================
    # 1. MACRO-RESOLUTION EXPERTS (Geometric / Low-Dimensional)
    # =========================================================================
    # Strategy: Use both LDA and QDA.
    # QDA is viable here because dimensionality is low (~12 features).
    # It captures class-specific covariance structures (heteroscedasticity).

    view = "macro"

    # LDA Experts (Macro)
    # We use 'lsqr' solver to support shrinkage.

    # Auto shrinkage (Ledoit-Wolf)
    experts[f"{view}_lda_auto"] = {
        "model": GenerativeExpert(
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        ),
        "view": view,
    }

    # Fixed Grid shrinkage
    for shrink in SHRINKAGE_GRID:
        experts[f"{view}_lda_shrink_{shrink}"] = {
            "model": GenerativeExpert(
                LinearDiscriminantAnalysis(solver="lsqr", shrinkage=shrink)
            ),
            "view": view,
        }

    # QDA Experts (Macro)
    # Regularized QDA to prevent overfitting even in low dimensions
    for reg in QDA_REG_GRID:
        experts[f"{view}_qda_reg_{reg}"] = {
            "model": GenerativeExpert(QuadraticDiscriminantAnalysis(reg_param=reg)),
            "view": view,
        }

    # =========================================================================
    # 2. MICRO-RESOLUTION EXPERTS (Texture, Shape, Margin / High-Dimensional)
    # =========================================================================
    # Strategy: Use only LDA with Shrinkage.
    # Dimensionality is high (192 features). QDA is likely to overfit or be singular.

    view = "micro"

    # Auto shrinkage
    experts[f"{view}_lda_auto"] = {
        "model": GenerativeExpert(
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        ),
        "view": view,
    }

    # Fixed Grid shrinkage
    for shrink in SHRINKAGE_GRID:
        experts[f"{view}_lda_shrink_{shrink}"] = {
            "model": GenerativeExpert(
                LinearDiscriminantAnalysis(solver="lsqr", shrinkage=shrink)
            ),
            "view": view,
        }

    # =========================================================================
    # 3. SYNERGISTIC EXPERTS (Concatenated / Highest-Dimensional)
    # =========================================================================
    # Strategy: Use only LDA with Shrinkage.
    # Captures cross-covariance between Macro and Micro features.

    view = "synergistic"

    # Auto shrinkage
    experts[f"{view}_lda_auto"] = {
        "model": GenerativeExpert(
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        ),
        "view": view,
    }

    # Fixed Grid shrinkage
    for shrink in SHRINKAGE_GRID:
        experts[f"{view}_lda_shrink_{shrink}"] = {
            "model": GenerativeExpert(
                LinearDiscriminantAnalysis(solver="lsqr", shrinkage=shrink)
            ),
            "view": view,
        }

    return experts
