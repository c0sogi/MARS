import os
import torch


class Config:
    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset of data
    DEBUG_SAMPLE_SIZE = 100  # Number of samples to use when DEBUG is True
    EXP_NAME = "idea_8"

    # =========================================================================
    # Directory Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    IMAGES_DIR = os.path.join(INPUT_DIR, "images")

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Working Directory for Outputs (Weights, Cache, Submission)
    WORKING_DIR = os.path.join("./working", EXP_NAME)

    # =========================================================================
    # Data & Target Settings
    # =========================================================================
    # Multi-Label Decomposition: We predict 'rust' and 'scab' independently.
    # The 'multiple_diseases' and 'healthy' classes are derived from these.
    TARGET_COLS = ["rust", "scab"]
    NUM_CLASSES = 2

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    N_FOLDS = 5
    EPOCHS = 20

    # Optimization
    LR = 1e-4
    MIN_LR = 1e-6
    WEIGHT_DECAY = 1e-4

    # Stochastic Weight Averaging (SWA)
    # Schedule: Standard Cosine Annealing for first 75%, then SWA.
    SWA_START_EPOCH = 15
    SWA_LR = 1e-5

    # Loss Settings
    USE_CLASS_WEIGHTS = True  # Apply inverse frequency weights for imbalance

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # =========================================================================
    # Model Specific Configurations
    # =========================================================================
    # We define a list of configs to allow the training pipeline to iterate
    # through or select specific models.
    MODEL_CONFIGS = [
        {
            "name": "tf_efficientnetv2_l.in21k_ft_in1k",
            "img_size": 480,
            "batch_size": 16,  # Adjusted for A100 40GB
            "dropout_rate": 0.3,
            "drop_path_rate": 0.2,
        },
        {
            "name": "convnext_base.fb_in22k_ft_in1k_384",
            "img_size": 384,
            "batch_size": 32,
            "dropout_rate": 0.3,
            "drop_path_rate": 0.2,
        },
    ]

    @classmethod
    def setup(cls):
        """
        Ensures the working directory exists.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        print(f"Config setup complete. Working directory: {cls.WORKING_DIR}")
        print(f"Device: {cls.DEVICE}")
