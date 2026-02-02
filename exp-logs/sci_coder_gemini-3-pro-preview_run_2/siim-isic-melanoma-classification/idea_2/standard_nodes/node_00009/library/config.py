import os
import torch


class Config:
    # ==============================
    # General Configuration
    # ==============================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SAMPLE_SIZE = 100  # Number of samples to use when DEBUG is True

    # ==============================
    # Directories & Paths
    # ==============================
    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for this specific idea (Deep Hybrid EfficientNet)
    # Caching and model checkpoints will be stored here
    WORK_DIR = "./working/idea_2"

    # Metadata Paths
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_ROOT, "sample_submission.csv")

    # Output Paths
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_CHECKPOINT_PATH = os.path.join(WORK_DIR, "model_best.pth")

    # ==============================
    # Data Configuration
    # ==============================
    IMG_SIZE = 224
    NUM_CLASSES = 1  # Binary classification (0=Benign, 1=Malignant)

    # ImageNet Normalization Constants
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # ==============================
    # Model Configuration
    # ==============================
    MODEL_NAME = "efficientnet_b0"
    PRETRAINED = True

    # Hybrid Fusion Head Configuration
    # The dimension of the metadata vector will be determined dynamically
    # based on the encoding (One-Hot + Numerical)
    FUSION_HIDDEN_DIM = 256
    DROPOUT_RATE = 0.3

    # ==============================
    # Training Configuration
    # ==============================
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2  # For AdamW

    # Scheduler Settings
    WARMUP_EPOCHS = 1

    # Focal Loss Hyperparameters
    # Alpha: Balancing factor (often set to inverse class frequency or tuned)
    # Gamma: Focusing parameter for hard examples
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 3
    EARLY_STOPPING_MIN_DELTA = 1e-4

    # ==============================
    # Hardware & Compute
    # ==============================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Optimized for 12 vCPUs

    @classmethod
    def setup(cls):
        """
        Creates necessary output directories.
        """
        os.makedirs(cls.WORK_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
