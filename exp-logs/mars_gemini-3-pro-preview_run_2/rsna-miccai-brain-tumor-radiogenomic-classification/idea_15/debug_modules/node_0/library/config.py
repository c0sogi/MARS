import os
import torch


class Config:
    """
    Configuration parameters for the Asymmetric Grouped EfficientNet pipeline.
    Implements settings for Modality-Specific Normalization and Integral-ROI Selection.
    """

    # ==========================================================================
    # PATHS & DIRECTORIES
    # ==========================================================================
    INPUT_DIR = "./input"
    TRAIN_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DIR = os.path.join(INPUT_DIR, "test")

    # Metadata Paths (Pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Working Directory (Writable)
    # Specific directory for Idea 15 to store caches and models
    WORKING_DIR = "./working/idea_15"
    CACHE_DIR = WORKING_DIR

    # Model Checkpoint
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Submission Output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================================================
    # REPRODUCIBILITY
    # ==========================================================================
    SEED = 42

    # ==========================================================================
    # DATA PIPELINE SETTINGS
    # ==========================================================================
    # Image Dimensions
    IMG_SIZE = (224, 224)

    # Modalities
    MODALITIES = ["FLAIR", "T1w", "T1wCE", "T2w"]

    # ROI Selection (Integral-Statistic Pipeline)
    # Search for anchor slice within 15%-85% of the volume depth
    ROI_DEPTH_MIN = 0.15
    ROI_DEPTH_MAX = 0.85
    # Moving average window for intensity profile smoothing
    ROI_SMOOTH_WINDOW = 5

    # Slice Extraction
    # Extract Anchor-5, Anchor, Anchor+5
    ROI_STRIDE = 5
    NUM_SLICES_PER_MODALITY = 3

    # Input Tensor Config
    # 4 Modalities * 3 Slices = 12 Channels
    TOTAL_INPUT_CHANNELS = len(MODALITIES) * NUM_SLICES_PER_MODALITY

    # ==========================================================================
    # MODEL HYPERPARAMETERS
    # ==========================================================================
    BACKBONE = "efficientnet_b0"
    NUM_CLASSES = 1

    # Modality-Specific Architecture
    # Ensures channels are isolated per modality in the stem
    CONV_GROUPS = 4
    # Normalizes statistics per modality group, not globally
    NORM_GROUPS = 4

    # Regularization
    DROPOUT_RATE = 0.5  # Dropout before final linear projection

    # ==========================================================================
    # TRAINING HYPERPARAMETERS
    # ==========================================================================
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-4  # Low LR to preserve pre-trained features
    WEIGHT_DECAY = 1e-2  # Aggressive weight decay
    PATIENCE = 3  # Early stopping patience

    # Hardware
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================================================
    # DEBUGGING
    # ==========================================================================
    DEBUG = False
    DEBUG_SIZE = 50  # Subset size for rapid debugging

    # ==========================================================================
    # UTILS
    # ==========================================================================
    @staticmethod
    def setup_directories():
        """Creates necessary working and submission directories."""
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
