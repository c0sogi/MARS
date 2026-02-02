import os
import torch
import numpy as np
import random


class Config:
    """
    Global configuration for the Pawpularity Tri-Paradigm Stacking Ensemble.
    Handles paths, model specifications, hyperparameters, and reproducibility settings.
    """

    # =========================================================================
    # General Configuration
    # =========================================================================
    PROJECT_NAME = "Pawpularity_TriParadigm_Ensemble"
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Use available CPUs for data loading
    NUM_WORKERS = 4

    # Debugging / Development Flags
    # Set DEBUG to True to run on a small subset of data for pipeline verification
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 100

    # =========================================================================
    # Directory & File Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for caching intermediate features (Idea 5)
    WORKING_DIR = "./working/idea_5"

    # Submission directory
    SUBMISSION_DIR = "./submission"

    # Ensure writeable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata File Paths (Pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Submission Path
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Configuration
    # =========================================================================
    TARGET_COL = "Pawpularity"
    ID_COL = "Id"
    PATH_COL = "file_path"

    # Binary Metadata Features provided in the dataset
    META_FEATURES = [
        "Subject Focus",
        "Eyes",
        "Face",
        "Near",
        "Action",
        "Accessory",
        "Group",
        "Collage",
        "Human",
        "Occlusion",
        "Info",
        "Blur",
    ]

    # =========================================================================
    # Model Configuration (Level-0 Experts)
    # =========================================================================
    # Defines the three orthogonal backbones for the ensemble.
    MODELS = {
        "clip": {
            "name": "openai/clip-vit-large-patch14",
            "library": "transformers",
            "batch_size": 32,
            "target_size": 224,
            "output_dim": 768,  # Projection dimension
        },
        "dinov2": {
            "name": "facebook/dinov2-large",
            "library": "transformers",
            "batch_size": 16,  # Large model requires smaller batch
            "target_size": 224,  # Standard ViT size
            "output_dim": 1024,  # Hidden size
        },
        "convnext": {
            "name": "convnext_large.fb_in22k_ft_in1k",
            "library": "timm",
            "batch_size": 32,
            "target_size": 224,
            "output_dim": 1536,  # Classifier input dimension
        },
    }

    # =========================================================================
    # Training Configuration
    # =========================================================================
    # Cross-Validation Settings
    N_FOLDS = 5

    # Ridge Regression Hyperparameters
    # Used for both Level-0 heads and Level-1 Meta-Learner
    # Log-spaced alphas to cover regularization strengths from 0.01 to 10000
    RIDGE_ALPHAS = np.logspace(-2, 4, 20).tolist()

    # =========================================================================
    # Utilities
    # =========================================================================
    @staticmethod
    def set_seed(seed=42):
        """
        Sets random seeds for Python, NumPy, and PyTorch to ensure reproducibility.

        Args:
            seed (int): The seed value to use.
        """
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


# Automatically set seed on import
Config.set_seed(Config.SEED)
