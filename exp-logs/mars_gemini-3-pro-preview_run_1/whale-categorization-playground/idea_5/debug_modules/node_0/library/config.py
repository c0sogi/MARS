import os
import torch


class Config:
    """
    Central configuration for the Whale Species Prediction task.
    Idea 5: DenseNet169 + Projection Head + ArcFace
    """

    # -------------------------------------------------------------------------
    # System & Reproducibility
    # -------------------------------------------------------------------------
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4  # Optimized for 12 vCPUs

    # -------------------------------------------------------------------------
    # File Paths & Directories
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Image Directories
    TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "train")
    TEST_IMG_DIR = os.path.join(INPUT_DIR, "test")

    # Metadata Files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Output Directories
    WORKING_DIR = "./working/idea_5"
    SUBMISSION_DIR = "./submission"

    # Ensure output directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Checkpoint & Submission Paths
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "model_best.pth")
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Data Configuration
    # -------------------------------------------------------------------------
    IMAGE_SIZE = 320  # 320x320 Resolution
    NUM_CLASSES = 4029  # Derived from metadata analysis

    # -------------------------------------------------------------------------
    # Model Architecture
    # -------------------------------------------------------------------------
    BACKBONE = "densenet169"
    EMBEDDING_DIM = 512  # Dimension of the projection head output
    DROPOUT_RATE = 0.5  # Dropout before projection

    # ArcFace Hyperparameters
    ARCFACE_MARGIN = 0.50
    ARCFACE_SCALE = 30.0

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 32  # Safe for A100 with 320x320 and DenseNet169
    NUM_EPOCHS = 25  # Sufficient for fine-tuning

    # Optimizer (AdamW)
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-4

    # Scheduler (Cosine Annealing)
    MIN_LR = 1e-6

    # -------------------------------------------------------------------------
    # Validation & Inference
    # -------------------------------------------------------------------------
    TOP_K = 5  # MAP@5
    USE_TTA = True  # Test Time Augmentation (Horizontal Flip)

    # -------------------------------------------------------------------------
    # Debugging / Development
    # -------------------------------------------------------------------------
    DEBUG = False  # Set to True to run on a subset
    DEBUG_SUBSET_SIZE = 100  # Number of samples to use in debug mode
