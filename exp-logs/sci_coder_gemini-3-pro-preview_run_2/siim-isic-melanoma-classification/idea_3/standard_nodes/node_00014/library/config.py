import os
import torch


class Config:
    """
    Configuration class for the Context-Gated Deep Hybrid Network (Idea 3).
    Centralizes all hyperparameters, file paths, and model settings.
    """

    # ==========================================
    # General Settings
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging

    # ==========================================
    # Directories and Paths
    # ==========================================
    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"

    # Output directories
    WORKING_DIR = "./working/idea_4"
    SUBMISSION_DIR = "./submission"

    # Create output directories if they don't exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata File Paths
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Submission Path
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Configuration
    # ==========================================
    # Image resolution increased to 256x256 as per strategy
    IMG_SIZE = 256
    IMG_CHANNELS = 3

    # ImageNet Normalization Constants
    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_STD = [0.229, 0.224, 0.225]

    # Metadata Features
    NUMERICAL_COLS = ["age_approx"]
    CATEGORICAL_COLS = ["sex", "anatom_site_general_challenge"]

    # ==========================================
    # Model Architecture
    # ==========================================
    MODEL_NAME = "efficientnet_b0"
    PRETRAINED = True
    NUM_CLASSES = 1

    # Dimension for the metadata projection in the context gating unit
    META_HIDDEN_DIM = 64

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 64
    EPOCHS = 15

    # Optimizer Settings (AdamW)
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2

    # Scheduler Settings (Cosine Annealing with Warmup)
    WARMUP_EPOCHS = 3

    # Loss Function Settings
    # Positive class weight calculated from EDA (Maj:Min ~ 55:1)
    POS_WEIGHT = 55.0

    # ==========================================
    # Hardware & Performance
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
    PIN_MEMORY = True
