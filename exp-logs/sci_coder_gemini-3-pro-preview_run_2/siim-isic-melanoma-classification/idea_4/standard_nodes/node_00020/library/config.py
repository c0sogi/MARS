import os
import torch


class Config:
    """
    Configuration class for the Multi-Task Hybrid EfficientNet-B1 solution.
    Centralizes all hyperparameters, file paths, model settings, and training configurations.
    """

    # =========================================================================
    # General Setup
    # =========================================================================
    SEED = 42
    # Flag to enable debugging mode (runs on a small subset of data)
    DEBUG = False
    # Number of samples to use when DEBUG is True
    DEBUG_SAMPLE_SIZE = 500

    # Compute settings
    NUM_WORKERS = 12  # Utilizing the available 12 vCPUs
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================================================================
    # File Paths & Directories
    # =========================================================================
    INPUT_ROOT = "./input"

    # Metadata paths (pre-generated in ./metadata directory)
    METADATA_DIR = "./metadata"
    TRAIN_META = os.path.join(METADATA_DIR, "train.csv")
    VAL_META = os.path.join(METADATA_DIR, "val.csv")
    TEST_META = os.path.join(METADATA_DIR, "test.csv")

    # Working directory for caching processed data and model checkpoints
    WORKING_DIR = "./working/idea_4"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Model checkpoint path
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "model_best.pth")

    # Submission output
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Configuration
    # =========================================================================
    # Input resolution scaled to 384x384 to capture fine-grained features
    IMG_SIZE = 384

    # Column definitions
    ID_COL = "image_name"
    TARGET_COL = "target"
    FILE_PATH_COL = "file_path"

    # Metadata Features for Hybrid Model
    # Categorical features to be One-Hot Encoded
    CAT_FEATURES = ["sex", "anatom_site_general_challenge"]
    # Numerical features to be Standardized
    NUM_FEATURES = ["age_approx"]

    # Auxiliary Task Target (Diagnosis)
    AUX_TARGET_COL = "diagnosis"

    # =========================================================================
    # Model Architecture
    # =========================================================================
    # Backbone: EfficientNet-B1 (unfrozen)
    MODEL_NAME = "efficientnet_b1"

    # Primary Head: Binary Classification (Malignant vs Benign)
    NUM_CLASSES = 1

    # Auxiliary Head: Multi-class Diagnosis Classification
    # Note: The exact number will be determined from the training data encoding,
    # but typically ISIC datasets have around 9 diagnostic categories.
    NUM_AUX_CLASSES = 9

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    # Batch size adjusted for A100 GPU (40GB) with 384x384 images
    BATCH_SIZE = 32
    EPOCHS = 10

    # Optimizer (AdamW)
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5

    # Loss Function Weights
    # Primary Loss (BCEWithLogitsLoss):
    # Positive weight set to ~55.0 to handle severe class imbalance (Maj:Min ~ 55:1)
    POS_WEIGHT = 55.0

    # Auxiliary Loss (CrossEntropyLoss):
    # Weight lambda = 0.1 to use diagnosis as a regularizer without dominating the gradient
    AUX_LOSS_WEIGHT = 0.1

    # Scheduler (Cosine Annealing with Warmup)
    WARMUP_EPOCHS = 1
    T_MAX = EPOCHS  # Cycle length for Cosine Annealing
    ETA_MIN = 1e-6  # Minimum learning rate
