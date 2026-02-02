import os
import torch


class Config:
    """
    Configuration class for Cassava Leaf Disease Classification.
    Centralizes settings for data, model, training curriculum, and inference.
    """

    # =========================================================================
    # General Configuration
    # =========================================================================
    PROJECT_NAME = "Cassava_Leaf_Disease_Classification_Idea_10"
    SEED = 42
    NUM_FOLDS = 5
    NUM_CLASSES = 5
    NUM_WORKERS = 12
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================================================================
    # Directory Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for caching intermediate files (e.g., processed data, checkpoints)
    WORKING_DIR = "./working/idea_10"

    # Output directory for the final submission file
    OUTPUT_DIR = "./submission"

    # Image Directories
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")

    # Metadata Files (Pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # =========================================================================
    # Model Configuration
    # =========================================================================
    # Backbone model name.
    # Corresponds to ConvNeXt Small pretrained on ImageNet-22k and fine-tuned on 1k.
    # Downstream scripts should map this string to the specific timm model name
    # (e.g., 'convnext_small.fb_in22k_ft_in1k').
    MODEL_BACKBONE = "convnext_small_in22k"

    # Model Architecture Settings
    DROP_PATH_RATE = 0.4  # Stochastic Depth rate
    USE_EMA = True  # Use Exponential Moving Average for weights
    EMA_DECAY = 0.9999  # EMA decay rate

    # =========================================================================
    # Training Configuration (Progressive Curriculum)
    # =========================================================================
    # General Training settings
    BATCH_SIZE = 32
    GRAD_ACCUM_STEPS = 1  # Effective batch size = BATCH_SIZE * GRAD_ACCUM_STEPS

    # Optimizer settings
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 0.05
    MAX_GRAD_NORM = 1.0
    LABEL_SMOOTHING = 0.1

    # Phase 1: Coarse Learning (Lower Resolution, High Regularization)
    # Focus on learning robust global features.
    PHASE1_EPOCHS = 12
    PHASE1_IMG_SIZE = 224
    PHASE1_MIXUP_PROB = 0.5
    PHASE1_CUTMIX_PROB = 0.5

    # Phase 2: Fine-Tuning (High Resolution, Regularization Annealing)
    # Focus on resolving fine-grained disease artifacts. MixUp/CutMix disabled.
    PHASE2_EPOCHS = 8
    PHASE2_IMG_SIZE = 384
    PHASE2_MIXUP_PROB = 0.0
    PHASE2_CUTMIX_PROB = 0.0

    # =========================================================================
    # Inference Configuration
    # =========================================================================
    TTA_FLIP = True  # Apply horizontal flip TTA during inference

    @classmethod
    def setup_directories(cls):
        """
        Ensures that the working and output directories exist.
        Should be called at the start of execution.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)

    @classmethod
    def get_cache_path(cls, filename):
        """
        Returns a safe path for caching files within the working directory.
        """
        cls.setup_directories()
        return os.path.join(cls.WORKING_DIR, filename)
