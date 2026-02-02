import numpy as np
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.naive_bayes import GaussianNB
from library.config import Config


class Expert:
    """
    A container for a specific model configuration within the ensemble.

    Attributes:
        name (str): Unique identifier for the expert.
        model (sklearn.base.BaseEstimator): The initialized model instance.
        view (str): The data view this expert consumes ('global', 'macro', or 'combined').
    """

    def __init__(self, name, model, view):
        self.name = name
        self.model = model
        self.view = view

    def __repr__(self):
        return f"Expert(name='{self.name}', view='{self.view}', model={self.model})"


class ModelFactory:
    """
    Factory class to generate the Precision-Generative Expert Library.
    Constructs experts across the Covariance Complexity Continuum.
    """

    @staticmethod
    def get_experts():
        """
        Generates the list of experts for the PCMRE strategy.

        Returns:
            list[Expert]: A list of configured Expert objects.
        """
        experts = []

        # =================================================================
        # Tier 1: The OAS-LDA Anchor (Theoretical Optimum)
        # =================================================================
        # Uses 'auto' shrinkage (Ledoit-Wolf/OAS approximation)
        # Views: Global and Combined
        for view in ["global", "combined"]:
            experts.append(
                Expert(
                    name=f"OAS_LDA_{view}",
                    model=LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
                    view=view,
                )
            )

        # =================================================================
        # Tier 2: The Fixed-Shrinkage Tuners (Empirical Exploitation)
        # =================================================================
        # Grid of fixed shrinkage values to fine-tune regularization
        # Views: Global and Combined
        for shrinkage in Config.LDA_FIXED_SHRINKAGE_GRID:
            # Format shrinkage string for cleaner names (e.g., 1e-05 -> 1e-5)
            shrink_str = (
                f"{shrinkage:.0e}".replace("0", "")
                if shrinkage < 0.01
                else str(shrinkage)
            )

            for view in ["global", "combined"]:
                experts.append(
                    Expert(
                        name=f"Fixed_LDA_{shrink_str}_{view}",
                        model=LinearDiscriminantAnalysis(
                            solver="lsqr", shrinkage=shrinkage
                        ),
                        view=view,
                    )
                )

        # =================================================================
        # Tier 3: The Macro-QDA Experts (Non-Linear Shape)
        # =================================================================
        # Class-Specific Covariance on low-dimensional physical features
        # View: Macro ONLY
        for reg_param in Config.QDA_REG_PARAM_GRID:
            experts.append(
                Expert(
                    name=f"Macro_QDA_reg{reg_param}",
                    model=QuadraticDiscriminantAnalysis(reg_param=reg_param),
                    view="macro",
                )
            )

        # =================================================================
        # Tier 4: The Diagonal Failsafe
        # =================================================================
        # High-bias, diagonal covariance anchor
        # View: Global
        experts.append(
            Expert(
                name="GNB_Global",
                model=GaussianNB(var_smoothing=Config.GNB_VAR_SMOOTHING),
                view="global",
            )
        )

        return experts
