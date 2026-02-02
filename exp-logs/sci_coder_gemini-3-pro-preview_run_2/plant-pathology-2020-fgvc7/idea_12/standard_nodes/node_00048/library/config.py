import os
import torch


class Config:
    """
    Configuration for Apple Disease Detection Task.
    Implements the strategy: Converged Fine-Grained Pathology Ensemble.

    Key Components:
    - Heterogeneous Ensemble: EfficientNetV2-L (480px) & ConvNeXt-Base (384px)
    - Structural Innovations: GeM Pooling, Multi-Sample Dropout
    - Training: 5-Fold CV, SWA, BCE with Inverse Class Weights
    - Inference: Rank-Calibrated Averaging with TTA
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    DEBUG = False  # Set to True for fast debugging runs on a subset
    DEBUG_SUBSET_SIZE = 100
    NUM_WORKERS = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================================================================
    # Directory Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for checkpoints and cached data
    WORKING_DIR = "./working/idea_12"

    # Submission directory
    SUBMISSION_DIR = "./submission"

    # Ensure output directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Data Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Image Directory
    IMAGES_DIR = os.path.join(INPUT_DIR, "images")

    # Final Submission File
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    FOLDS = 5
    EPOCHS = 35

    # Learning Rate Strategy
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 1e-2
    MAX_GRAD_NORM = 1000.0

    # Stochastic Weight Averaging (SWA)
    USE_SWA = True
    SWA_START_EPOCH = 25
    SWA_LR = 1e-5

    # Loss & Regularization
    LABEL_SMOOTHING = 0.05

    # =========================================================================
    # Model Architecture & Ensemble
    # =========================================================================
    # Target Strategy: Multi-Label Decomposition
    # We predict 2 binary targets: [Is_Rust, Is_Scab]
    # Healthy = (1-Rust)*(1-Scab)
    # Multiple = Rust*Scab
    NUM_TARGETS = 2

    # Heterogeneous Ensemble Definition
    # Each dictionary defines a specific backbone configuration
    MODELS = [
        {
            "name": "tf_efficientnetv2_l.in21k_ft_in1k",
            "img_size": 480,
            "batch_size": 4,  # Reduced to fit 16GB VRAM (Effective batch = 4 * 4 = 16)
            "gem_p": 3.0,  # Initial p for GeM Pooling
            "dropout": 0.2,
            "num_msd": 5,  # Number of Multi-Sample Dropout branches
            "msd_dropout": 0.2,
        },
        {
            "name": "convnext_base.fb_in22k_ft_in1k_384",
            "img_size": 384,
            "batch_size": 16,  # Reduced for safety (Effective batch = 16 * 4 = 64)
            "gem_p": 3.0,
            "dropout": 0.2,
            "num_msd": 5,
            "msd_dropout": 0.2,
        },
    ]

    # =========================================================================
    # Inference Configuration
    # =========================================================================
    TTA_FLIP = True  # Enable Horizontal Flip Test Time Augmentation

    # Final Submission Columns (Required by competition)
    OUTPUT_COLS = ["healthy", "multiple_diseases", "rust", "scab"]

    @classmethod
    def print_config(cls):
        """Prints the current configuration for verification."""
        print("=" * 40)
        print(f"CONFIGURATION (Seed: {cls.SEED})")
        print("=" * 40)
        print(f"Device:       {cls.DEVICE}")
        print(f"Folds:        {cls.FOLDS}")
        print(f"Epochs:       {cls.EPOCHS} (SWA start: {cls.SWA_START_EPOCH})")
        print(f"Working Dir:  {cls.WORKING_DIR}")
        print("-" * 40)
        print(f"Ensemble Models ({len(cls.MODELS)}):")
        for i, m in enumerate(cls.MODELS):
            print(f"  {i+1}. {m['name']}")
            print(
                f"     Size: {m['img_size']}x{m['img_size']} | Batch: {m['batch_size']}"
            )
            print(f"     GeM p: {m['gem_p']} | MSD: {m['num_msd']}x{m['msd_dropout']}")
        print("=" * 40)
