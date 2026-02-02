import os
import torch


class Config:
    # -------------------------------------------------------------------------
    # General Configuration
    # -------------------------------------------------------------------------
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SAMPLE_SIZE = 500  # Number of samples to use when DEBUG is True

    # -------------------------------------------------------------------------
    # Compute
    # -------------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 12  # Utilizing available vCPUs

    # -------------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "train")
    TEST_IMG_DIR = os.path.join(INPUT_DIR, "test")

    METADATA_DIR = "./metadata"
    TRAIN_META = os.path.join(METADATA_DIR, "train.csv")
    VAL_META = os.path.join(METADATA_DIR, "val.csv")
    TEST_META = os.path.join(METADATA_DIR, "test.csv")

    ANNOTATIONS_PATH = os.path.join(INPUT_DIR, "train_annotations.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Directories
    WORKING_DIR = "./working/idea_2"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Checkpoint Path
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # -------------------------------------------------------------------------
    # Model Hyperparameters
    # -------------------------------------------------------------------------
    BACKBONE = "tf_efficientnetv2_s"
    IMG_SIZE = 512
    NUM_CLASSES = 11

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 16
    EPOCHS = 10
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-2
    MAX_GRAD_NORM = 10.0

    # Scheduler
    T_MAX = EPOCHS  # For CosineAnnealingLR
    MIN_LR = 1e-6

    # -------------------------------------------------------------------------
    # Auxiliary Segmentation Task
    # -------------------------------------------------------------------------
    USE_AUX = True
    AUX_LOSS_WEIGHT = 0.5  # Weight for the segmentation loss

    # -------------------------------------------------------------------------
    # Target Labels
    # -------------------------------------------------------------------------
    LABELS = [
        "ETT - Abnormal",
        "ETT - Borderline",
        "ETT - Normal",
        "NGT - Abnormal",
        "NGT - Borderline",
        "NGT - Incompletely Imaged",
        "NGT - Normal",
        "CVC - Abnormal",
        "CVC - Borderline",
        "CVC - Normal",
        "Swan Ganz Catheter Present",
    ]

    @classmethod
    def setup(cls):
        """Creates necessary output directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize directories on import
Config.setup()
