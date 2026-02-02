import numpy as np
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.covariance import OAS
from library.config import Config
from library.utils import setup_logger

# Initialize logger
logger = setup_logger("models")


class ExpertLibrary:
    """
    Factory class to construct the pool of classifiers for the
    Orthogonal-Basis Hybrid-Complexity Ensemble.

    This library provides two tiers of experts:
    1. Tier 1: LDA models with various shrinkage strategies for high-dimensional Global features.
    2. Tier 2: QDA models with regularization for low-dimensional Zernike features.
    """

    def __init__(self):
        self.lda_strategies = Config.LDA_SHRINKAGE_STRATEGIES
        self.qda_reg_params = Config.QDA_REG_PARAMS

    def get_tier1_experts(self):
        """
        Constructs Tier 1 experts: Linear Discriminant Analysis on Global Features.
        Uses various shrinkage strategies to handle high dimensionality (N ~ P).

        Strategies supported:
        - 'ledoit_wolf': Uses sklearn's 'auto' shrinkage (Ledoit-Wolf lemma).
        - 'oas': Uses Oracle Approximating Shrinkage via covariance_estimator.
        - float: Uses a fixed shrinkage coefficient.

        Returns:
            dict: Dictionary of {name: estimator}
        """
        experts = {}
        logger.info(
            f"Constructing Tier 1 (LDA) experts with strategies: {self.lda_strategies}"
        )

        for strategy in self.lda_strategies:
            if strategy == "ledoit_wolf":
                # 'auto' shrinkage in LDA uses Ledoit-Wolf lemma
                name = "LDA_LedoitWolf"
                # Solver 'lsqr' is required for shrinkage
                clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")

            elif strategy == "oas":
                # Use Oracle Approximating Shrinkage as the covariance estimator
                name = "LDA_OAS"
                # covariance_estimator requires solver='lsqr' or 'eigen'
                # This overrides the standard LDA covariance calculation
                clf = LinearDiscriminantAnalysis(
                    solver="lsqr", covariance_estimator=OAS()
                )

            elif isinstance(strategy, (float, int)):
                # Fixed shrinkage coefficient
                name = f"LDA_Fixed_{strategy}"
                clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=strategy)

            else:
                logger.warning(f"Unknown LDA strategy: {strategy}. Skipping.")
                continue

            experts[name] = clf

        return experts

    def get_tier2_experts(self):
        """
        Constructs Tier 2 experts: Quadratic Discriminant Analysis on Zernike Features.
        Uses regularization to control covariance estimation in lower dimensions.

        Returns:
            dict: Dictionary of {name: estimator}
        """
        experts = {}
        logger.info(
            f"Constructing Tier 2 (QDA) experts with reg_params: {self.qda_reg_params}"
        )

        for reg_param in self.qda_reg_params:
            name = f"QDA_Reg_{reg_param}"
            # QDA allows for a regularization parameter to regularize the covariance estimate
            clf = QuadraticDiscriminantAnalysis(reg_param=reg_param)
            experts[name] = clf

        return experts
