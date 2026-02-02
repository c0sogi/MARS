import os
import torch


class Config:
    # ==========================================
    # Reproducibility & System
    # ==========================================
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4  # Optimized for 12 vCPUs

    # ==========================================
    # Data Configuration
    # ==========================================
    IMG_SIZE = 256
    BATCH_SIZE = 32
    N_FOLDS = 5
    N_CLASSES = 4
    CLASS_LABELS = ["healthy", "multiple_diseases", "rust", "scab"]

    # Debugging
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 100  # Number of samples to use when DEBUG is True

    # ==========================================
    # Path Configuration
    # ==========================================
    # Input Directories (Read-Only)
    INPUT_DIR = "./input"
    IMAGES_DIR = os.path.join(INPUT_DIR, "images")

    # Metadata (Pre-generated Stratified Splits)
    METADATA_DIR = "./metadata"
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Original Data Files
    TRAIN_CSV_PATH = os.path.join(INPUT_DIR, "train.csv")
    TEST_CSV_PATH = os.path.join(INPUT_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Directories (Writable)
    WORKING_DIR = "./working/idea_6"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Model Architecture
    # ==========================================
    BACKBONE = "resnet34"
    PRETRAINED = True

    # Generalized Mean (GeM) Pooling
    GEM_P = 3.0
    GEM_LEARNABLE = True

    # Multi-Sample Dropout Head
    USE_MULTI_SAMPLE_DROPOUT = True
    DROPOUT_RATES = [0.5, 0.5, 0.5, 0.5, 0.5]

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2
    MAX_GRAD_NORM = 1000

    # Cosine Annealing Scheduler
    T_0 = 50  # Cycle length matches EPOCHS
    T_MULT = 1
    MIN_LR = 1e-6

    # Loss Function
    USE_CLASS_WEIGHTS = True  # To handle class imbalance

    # ==========================================
    # Regularization & Augmentation
    # ==========================================
    # Mixup / CutMix
    MIXUP_ALPHA = 0.4
    CUTMIX_ALPHA = 1.0
    MIXUP_PROB = (
        0.0  # Disabled to improve convergence on short budget (Cite Lesson 00011)
    )

    @classmethod
    def setup_directories(cls):
        """Creates necessary working and submission directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
