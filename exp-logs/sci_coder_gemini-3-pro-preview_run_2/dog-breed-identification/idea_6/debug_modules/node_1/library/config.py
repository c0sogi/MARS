import os
import torch


class Config:
    """
    Global configuration for the Dual-Stream Multi-View Heterogeneous Ensemble.
    """

    # --- Directories ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_6"
    SUBMISSION_DIR = "./submission"

    # --- File Paths ---
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --- Global Settings ---
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # 12 vCPUs available, leaving some overhead
    NUM_WORKERS = 4

    # --- Stream A: CNN (ConvNeXt-Large) ---
    # Uses torchvision "New Recipe" weights (IMAGENET1K_V1)
    STREAM_A = {
        "name": "convnext_large",
        "weights": "IMAGENET1K_V1",
        "batch_size": 64,  # A100 40GB can handle this for ConvNeXt
        "input_size": 224,  # Standard resolution for V1 weights
        "embedding_dim": 1536,
        "cache_prefix": "stream_a_convnext",
    }

    # --- Stream B: Transformer (ViT-H-14) ---
    # Uses SWAG weights (IMAGENET1K_SWAG_E2E_V1)
    STREAM_B = {
        "name": "vit_h_14",
        "weights": "IMAGENET1K_SWAG_E2E_V1",
        "batch_size": 16,  # ViT-H is massive, conservative batch size
        "input_size": 518,  # SWAG weights typically use 518x518
        "embedding_dim": 1280,
        "cache_prefix": "stream_b_vit",
    }

    # --- Multi-View Strategy ---
    # Local view zoom factor (1.5x larger before crop)
    LOCAL_VIEW_SCALE = 1.5

    # --- Classifier Settings ---
    # LogisticRegressionCV parameters
    LOGREG_PARAMS = {
        "Cs": 10,  # Number of C values to try in log-space
        "cv": 5,  # 5-fold cross-validation
        "solver": "saga",
        "max_iter": 2000,
        "n_jobs": 1,
        "random_state": SEED,
    }

    @classmethod
    def setup(cls):
        """Ensures that working and submission directories exist."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize directories on import
Config.setup()
