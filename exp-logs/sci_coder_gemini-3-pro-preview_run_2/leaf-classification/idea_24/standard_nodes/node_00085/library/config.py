import os


class Config:
    """
    Configuration module for the Regularized Quadratic-Linear Generative Ensemble (RQLGE).
    Defines global constants, file paths, and hyperparameter grids for the expert library.
    """

    # =========================================================================
    # PATHS & DIRECTORIES
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Cache directory for deterministic data processing (e.g., morphological features)
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_24")

    # Submission directory
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Metadata File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Image Directory (relative to INPUT_DIR)
    IMAGES_DIR = "images"

    # =========================================================================
    # GLOBAL SETTINGS
    # =========================================================================
    RANDOM_SEED = 42

    # Data Processing Requirements
    FLOAT_PRECISION = "float64"  # Double precision for density estimation
    SCALER_METHOD = "yeo-johnson"  # PowerTransformer method for Gaussianization

    # =========================================================================
    # EXPERT LIBRARY HYPERPARAMETERS
    # =========================================================================

    # Group A: Linear Anchors (LDA)
    # Shrinkage values: 'auto' (Ledoit-Wolf) and fixed grid.
    # Note: OAS is handled as a specific model type.
    # Cite solution_lesson_node_00081: Treat hyperparameter variations as distinct ensemble candidates.
    LDA_SHRINKAGE_GRID = [
        "auto",
        0.0001,
        0.001,
        0.01,
        0.05,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    ]

    # Group B: Quadratic Innovators (Regularized QDA)
    # Cite solution_lesson_node_00072: Prefer LDA over non-parametric or complex models when Gaussianized.
    # QDA Removed to focus on robust LDA experts.
    QDA_REG_PARAM_GRID = []

    # Group C: Morphological Experts
    # Uses LDA with Ledoit-Wolf shrinkage on morphological features
    MORPH_MODEL_TYPE = "LDA"
    MORPH_SHRINKAGE = "auto"

    def __init__(self, debug=False):
        """
        Initialize configuration.

        Args:
            debug (bool): If True, enables debug mode (e.g., smaller datasets).
        """
        self.debug = debug
        self._ensure_directories()

    def _ensure_directories(self):
        """Creates necessary working and submission directories."""
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        os.makedirs(self.SUBMISSION_DIR, exist_ok=True)

    def get_expert_library_config(self):
        """
        Generates the configuration list for all experts in the ensemble library.

        Returns:
            list[dict]: A list of dictionaries, where each dict contains:
                - 'name': Unique identifier string
                - 'type': Model type ('LDA', 'LDA_OAS', 'QDA')
                - 'view': Feature view ('global', 'morphological')
                - 'params': Dictionary of model hyperparameters
        """
        experts = []

        # --- Group A: Linear Anchors (Global View) ---
        # 1. LDA with Ledoit-Wolf ('auto') and Fixed Shrinkage values
        for shrinkage in self.LDA_SHRINKAGE_GRID:
            shrink_name = str(shrinkage)
            experts.append(
                {
                    "name": f"LDA_Global_Shrinkage_{shrink_name}",
                    "type": "LDA",
                    "view": "global",
                    "params": {"solver": "lsqr", "shrinkage": shrinkage},
                }
            )

        # 2. LDA with OAS Shrinkage (Oracle Approximating Shrinkage)
        experts.append(
            {
                "name": "LDA_Global_OAS",
                "type": "LDA_OAS",
                "view": "global",
                "params": {},  # OAS implementation handles its own parameters
            }
        )

        # --- Group B: Combined Experts (Global + Morphological) ---
        # Cite solution_lesson_node_00063: Prefer concatenating distinct feature sets into a single global matrix.
        for shrinkage in self.LDA_SHRINKAGE_GRID:
            shrink_name = str(shrinkage)
            experts.append(
                {
                    "name": f"LDA_Combined_Shrinkage_{shrink_name}",
                    "type": "LDA",
                    "view": "combined",
                    "params": {"solver": "lsqr", "shrinkage": shrinkage},
                }
            )

        # LDA with OAS Shrinkage on Combined View
        experts.append(
            {
                "name": "LDA_Combined_OAS",
                "type": "LDA_OAS",
                "view": "combined",
                "params": {},
            }
        )

        # --- Group C: Orthogonal Morphometric Experts (Morphological View) ---
        # LDA with Ledoit-Wolf on purely morphological features
        experts.append(
            {
                "name": "LDA_Morph_LedoitWolf",
                "type": "LDA",
                "view": "morphological",
                "params": {"solver": "lsqr", "shrinkage": self.MORPH_SHRINKAGE},
            }
        )

        return experts
