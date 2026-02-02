import os


class Config:
    """
    Global configuration for the Balanced Projected-Context Dual-Stream Network (BPCDS-Net).
    """

    # =========================================================================
    # Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory specific to this iteration (Idea 45)
    WORKING_DIR = "./working/idea_45"

    # Sub-directories for artifacts
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"

    # Ensure working directories exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # Reproducibility
    # =========================================================================
    SEED = 42

    # =========================================================================
    # Data Preprocessing
    # =========================================================================
    # Radiological Windowing (Lung Window)
    WINDOW_LEVEL = -600
    WINDOW_WIDTH = 1500

    # Image Dimensions
    # EfficientNet-B2 native resolution is 260x260
    IMG_SIZE = 260

    # Slice Selection
    # 1 Anchor (Max Lung Area) + 2 Boundaries
    NUM_SLICES = 3

    # =========================================================================
    # Model Architecture (BPCDS-Net)
    # =========================================================================
    BACKBONE_NAME = "efficientnet_b2"

    # Dimensionality
    LATENT_DIM = 64  # Bottleneck projection size
    HIDDEN_DIM = 128  # MLP hidden layer size

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 32
    EPOCHS = 30

    # Optimization
    # Differential Learning Rates
    LR_BACKBONE = 1e-4
    LR_HEADS = 5e-4

    WEIGHT_DECAY = 0.01

    # Scheduler (Cosine Annealing)
    T_MAX = EPOCHS
    ETA_MIN = 1e-6

    # =========================================================================
    # Inference & Metrics
    # =========================================================================
    # Post-processing clip for submission (ml)
    SIGMA_MIN_CLIP = 70

    # =========================================================================
    # Debugging
    # =========================================================================
    # Set to True to run on a small subset of data
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 50
