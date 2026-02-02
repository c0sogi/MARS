import os
import torch


class Config:
    # =========================================================================
    # Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for the specific idea implementation
    WORKING_DIR = "./working/idea_7"

    # Sub-directories for artifacts
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"

    # Metadata Files (Pre-generated)
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_meta.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val_meta.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_meta.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # =========================================================================
    # Data Configuration
    # =========================================================================
    # Image Dimensions
    IMG_SIZE = 640  # Resolution for EfficientNet-B0 detection

    # Class Definitions
    # 0-13 are findings, 14 is "No finding"
    NUM_CLASSES = 14
    CLASS_ID_NO_FINDING = 14

    CLASS_MAP = {
        0: "Aortic enlargement",
        1: "Atelectasis",
        2: "Calcification",
        3: "Cardiomegaly",
        4: "Consolidation",
        5: "ILD",
        6: "Infiltration",
        7: "Lung Opacity",
        8: "Nodule/Mass",
        9: "Other lesion",
        10: "Pleural effusion",
        11: "Pleural thickening",
        12: "Pneumothorax",
        13: "Pulmonary fibrosis",
        14: "No finding",
    }

    # Data Loading
    NUM_WORKERS = 4
    PIN_MEMORY = True

    # Preprocessing
    INVERT_MONOCHROME1 = True  # Semantic Normalization

    # Augmentation
    MIN_VISIBILITY = 0.3  # Geometric augmentation constraint

    # =========================================================================
    # Model Architecture
    # =========================================================================
    BACKBONE = "tf_efficientnet_b0_ns"
    PRETRAINED = True

    # Neck / BiFPN
    NECK_CHANNELS = 64

    # Heads
    # Initialization std for regression heads to prevent exploding gradients
    HEAD_INIT_STD = 0.001

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    NUM_EPOCHS = 20
    BATCH_SIZE = 16  # Adjusted for A100 memory with 640x640 and gradients

    # Optimizer (AdamW)
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Scheduler (Cosine Annealing)
    T_MAX = NUM_EPOCHS
    ETA_MIN = 1e-6

    # Loss Weights
    LOSS_WEIGHT_HEATMAP = 1.0
    LOSS_WEIGHT_SIZE = 0.1
    LOSS_WEIGHT_OFFSET = 0.1
    LOSS_WEIGHT_GLOBAL = 1.0  # Auxiliary head

    # =========================================================================
    # Inference & Post-processing
    # =========================================================================
    GLOBAL_THRESHOLD = 0.8  # Threshold for "No finding" gate
    CONF_THRESHOLD = 0.01  # Confidence threshold for bounding boxes
    IOU_THRESHOLD = 0.4  # IoU threshold for NMS/WBF

    # =========================================================================
    # Setup Logic
    # =========================================================================
    @classmethod
    def setup(cls):
        """Creates necessary directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Run setup immediately when module is imported
Config.setup()
