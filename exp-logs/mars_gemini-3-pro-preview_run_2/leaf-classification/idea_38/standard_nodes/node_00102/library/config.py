import os
import numpy as np


class Config:
    """
    Global configuration for the Multi-Resolution Gaussianized Dynamic Ensemble (MRGDE).
    Defines paths, constants, hyperparameters, and the expert library structure.
    """

    # =========================================================================
    # 1. Environment & Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_38"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Create necessary directories
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # 2. Global Settings
    # =========================================================================
    RANDOM_SEED = 42
    # Strictly enforce double precision to minimize numerical noise at metric floor
    FLOAT_PRECISION = np.float64

    # Dataset Columns
    ID_COL = "id"
    TARGET_COL = "species"
    IMAGE_PATH_COL = "image_path"

    # =========================================================================
    # 3. Hyperparameters
    # =========================================================================

    # --- Dimension A: Gaussian Resolution Spectrum (Preprocessing) ---
    BASIS_PARAMETRIC = "yeo-johnson"  # PowerTransformer
    BASIS_ROBUST = "quantile_robust"  # QuantileTransformer (n=20)
    BASIS_FLEXIBLE = "quantile_flexible"  # QuantileTransformer (n=100)

    # Quantile Hyperparameters
    N_QUANTILES_ROBUST = 20
    N_QUANTILES_FLEXIBLE = 100

    # --- Dimension B: Estimators ---
    MODEL_LDA_OAS = "lda_oas"  # LDA with OAS Covariance
    MODEL_LDA_FIXED = "lda_fixed"  # LDA with Fixed Shrinkage
    MODEL_LOGREG = "logreg_cv"  # Logistic Regression CV (L2)

    # Estimator Hyperparameters
    LDA_FIXED_SHRINKAGE = 0.01
    LOGREG_MAX_ITER = 1000

    # --- Dimension C: Feature Views ---
    VIEW_GLOBAL = "global"  # Provided 192 features
    VIEW_COMBINED = "combined"  # Global + Morphometrics (Hu Moments, etc.)

    # =========================================================================
    # 4. Expert Library Generation
    # =========================================================================
    @staticmethod
    def get_expert_library():
        """
        Generates the list of expert configurations for the ensemble.

        Logic:
        1. Generative Experts (LDA-OAS, LDA-Fixed):
           - Applied to ALL Bases (Parametric, Robust, Flexible).
           - Applied to ALL Views (Global, Combined).

        2. Discriminative Anchor (LogReg):
           - Applied ONLY to the Parametric Basis (Regularization constraint).
           - Applied to ALL Views.

        Returns:
            list[dict]: A list of configuration dictionaries, each defining a unique expert.
        """
        experts = []

        # 1. Generative Experts (LDA)
        lda_models = [Config.MODEL_LDA_OAS, Config.MODEL_LDA_FIXED]
        all_bases = [
            Config.BASIS_PARAMETRIC,
            Config.BASIS_ROBUST,
            Config.BASIS_FLEXIBLE,
        ]
        all_views = [Config.VIEW_GLOBAL, Config.VIEW_COMBINED]

        for model in lda_models:
            for basis in all_bases:
                for view in all_views:
                    experts.append(
                        {
                            "model": model,
                            "basis": basis,
                            "view": view,
                            "id": f"{model}_{basis}_{view}",
                        }
                    )

        # 2. Discriminative Anchor (LogReg)
        # Constraint: Parametric Basis only to prevent overfitting on quantile noise
        logreg_model = Config.MODEL_LOGREG
        parametric_basis = Config.BASIS_PARAMETRIC

        for view in all_views:
            experts.append(
                {
                    "model": logreg_model,
                    "basis": parametric_basis,
                    "view": view,
                    "id": f"{logreg_model}_{parametric_basis}_{view}",
                }
            )

        return experts
