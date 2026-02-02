import os
import torch


class Config:
    """
    Global configuration for the Hotel Identification task.
    Implements the Curriculum-Based EfficientNet Retrieval System parameters.
    """

    # ==========================================
    # System & File Paths
    # ==========================================
    # Base directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_3"
    SUBMISSION_DIR = "./submission"

    # Ensure output directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata files (Pre-generated)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Output files
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Hyperparameters
    # ==========================================
    IMG_SIZE = 224  # EfficientNet-B0 input resolution
    BATCH_SIZE = 64  # Small batch size for regularization (Lesson 4)
    NUM_WORKERS = 4  # Number of data loading workers
    NUM_CLASSES = 7770  # Total unique hotels (from EDA)

    # Debugging / Development
    DEBUG = False  # Set to True to run on a small subset
    DEBUG_SAMPLE_SIZE = 2000

    # ==========================================
    # Model Architecture
    # ==========================================
    BACKBONE_NAME = "efficientnet_b0"
    EMBEDDING_DIM = 512  # Dimension for the GeM/Neck output

    # ArcFace Head Parameters
    MARGIN = 0.50  # Angular margin
    SCALE = 30.0  # Scaling factor

    # ==========================================
    # Training Curriculum
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Optimization
    LR = 1e-3
    WEIGHT_DECAY = 1e-4

    # Curriculum Schedule
    # Stage 1: Softmax Warmup (Plasticity Preservation)
    WARMUP_EPOCHS = 1

    # Stage 2: Metric Fine-Tuning (ArcFace)
    FINE_TUNE_EPOCHS = 10

    # Total Training Duration
    TOTAL_EPOCHS = WARMUP_EPOCHS + FINE_TUNE_EPOCHS

    # Validation & Inference
    TOP_K = 5  # For MAP@5 metric
    PATIENCE = 3  # Early stopping patience
