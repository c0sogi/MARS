import os
import torch


class Config:
    """
    Configuration class for Cassava Leaf Disease Classification.
    Implements parameters for a Dynamic Fidelity Curriculum with Synchronized EMA.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    EXPERIMENT_NAME = "idea_13"
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    NUM_WORKERS = 12
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================================================================
    # Directory Paths
    # =========================================================================
    # Input paths (Read-Only)
    INPUT_DIR = "./input"
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")

    # Metadata paths (Pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Output paths (Working directory)
    WORKING_DIR = os.path.join("./working", EXPERIMENT_NAME)
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    LOG_DIR = os.path.join(WORKING_DIR, "logs")
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Model Architecture
    # =========================================================================
    MODEL_NAME = "convnext_small.fb_in1k"  # Pre-trained on ImageNet-1k
    NUM_CLASSES = 5
    DROP_PATH_RATE = 0.4  # Stochastic Depth rate
    USE_EMA = True
    EMA_DECAY = 0.9999

    # =========================================================================
    # Training Strategy (5-Fold Stratified CV)
    # =========================================================================
    NUM_FOLDS = 5

    # Optimization
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2
    MIN_LR = 1e-6

    # =========================================================================
    # Dynamic Fidelity Curriculum (Two Phases)
    # =========================================================================

    # --- Phase 1: Coarse Feature Learning ---
    # Focus: Global features, heavy regularization, lower resolution for speed
    PHASE_1_EPOCHS = 7
    PHASE_1_IMG_SIZE = 224
    PHASE_1_BATCH_SIZE = 32
    PHASE_1_ACCUM_STEPS = 1  # Effective Batch Size = 32
    PHASE_1_MIXUP_PROB = 0.5
    PHASE_1_LABEL_SMOOTHING = 0.0  # Use Soft Target Cross Entropy for Mixup

    # --- Phase 2: Fine-Grained Refinement ---
    # Focus: Fine details, reduced regularization, high resolution
    PHASE_2_EPOCHS = 5
    PHASE_2_IMG_SIZE = 384
    PHASE_2_BATCH_SIZE = 16  # Reduced due to memory constraints of larger image
    PHASE_2_ACCUM_STEPS = 2  # Effective Batch Size = 32 (16 * 2)
    PHASE_2_MIXUP_PROB = 0.0  # Disable Mixup to refine on real images
    PHASE_2_LABEL_SMOOTHING = 0.1

    # Total Epochs per Fold
    TOTAL_EPOCHS = PHASE_1_EPOCHS + PHASE_2_EPOCHS

    # =========================================================================
    # Inference
    # =========================================================================
    TTA_FLIP = True  # Test Time Augmentation: Horizontal Flip

    @classmethod
    def setup(cls):
        """
        Ensures all necessary output directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Print configuration summary
        print(f"Configuration Initialized for Experiment: {cls.EXPERIMENT_NAME}")
        print(f"Output Directory: {cls.WORKING_DIR}")
        print(f"Device: {cls.DEVICE}")


# Initialize directories on import
Config.setup()
