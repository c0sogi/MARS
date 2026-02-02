import os
import torch


class Config:
    """
    Configuration for Idea 9: Swin Transformer + DyHead + ATSS + Query Classifier.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    EXP_NAME = "idea_9"
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset of data for debugging
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # =========================================================================
    # Directories & Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for checkpoints, logs, and cached data
    WORKING_DIR = os.path.join("./working", EXP_NAME)

    # Submission directory
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Metadata Paths (Pre-generated)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # =========================================================================
    # Data Preprocessing
    # =========================================================================
    IMG_SIZE = 640  # Target size for Letterbox Resizing (Longest dimension)

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================
    # Backbone: Swin Transformer Tiny
    # We use 'swin_tiny_patch4_window7_224' from timm.
    # Expected feature channels from stages 2, 3, 4: [192, 384, 768]
    BACKBONE = "swin_tiny_patch4_window7_224"
    BACKBONE_OUT_CHANNELS = [192, 384, 768]

    # Dynamic Head (DyHead) Settings
    DYHEAD_CHANNELS = 256
    DYHEAD_NUM_BLOCKS = 6  # Number of DyHead blocks

    # Heads
    NUM_CLASSES_DET = 1  # Object Detection: Opacity
    NUM_CLASSES_STUDY = (
        4  # Study Classification: Negative, Typical, Indeterminate, Atypical
    )

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 8  # Tuned for A100 40GB
    EPOCHS = 15

    # Optimizer (AdamW)
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 0.05

    # Scheduler (Cosine Annealing)
    MIN_LR = 1e-6
    WARMUP_EPOCHS = 1

    # Loss Weights
    LOSS_WEIGHT_CLS_DET = 1.0  # Detection Classification (GFL)
    LOSS_WEIGHT_BOX = 2.0  # Bounding Box Regression (GIoU)
    LOSS_WEIGHT_STUDY = 1.0  # Study Classification (CrossEntropy)

    # =========================================================================
    # Inference & Post-processing
    # =========================================================================
    CONF_THRESHOLD = 0.001  # Confidence threshold for keeping boxes
    IOU_THRESHOLD = 0.6  # IoU threshold for NMS
    WBF_IOU_THRESHOLD = 0.5  # IoU threshold for Weighted Boxes Fusion

    # Study Labels Mapping
    STUDY_ID_MAP = {
        0: "Negative for Pneumonia",
        1: "Typical Appearance",
        2: "Indeterminate Appearance",
        3: "Atypical Appearance",
    }

    @classmethod
    def setup(cls):
        """Ensures necessary directories exist."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize environment
Config.setup()
