import os
import torch
import numpy as np
import random


def seed_everything(seed=42):
    """Sets the random seed for reproducibility across all libraries."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Config:
    # --- General Configuration ---
    DEBUG = False
    DEBUG_DATA_SIZE = 20  # Number of studies to use when DEBUG is True
    SEED = 42
    PROJECT_NAME = "RSNA_Cervical_Spine_Fracture_Detection"

    # --- Path Configuration ---
    ROOT_DIR = "./input"
    TRAIN_IMAGES_DIR = os.path.join(ROOT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(ROOT_DIR, "test_images")

    # Metadata (Pre-generated split files)
    METADATA_DIR = "./metadata"
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(ROOT_DIR, "sample_submission.csv")

    # Output Directories
    # Using 'idea_16' to isolate this specific solution's artifacts
    WORKING_DIR = os.path.join("./working", "idea_16")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --- Data Preprocessing ---
    # 2.5D Stacking: Input is a stack of 3 slices (z-1, z, z+1)
    IN_CHANNELS = 3
    # Image resolution (Using 384 for balance between B4 requirements and memory)
    IMAGE_SIZE = (384, 384)
    # Sequence length: Number of slices sampled per study for the LSTM
    SEQ_LEN = 96

    # --- Model Architecture ---
    BACKBONE = "tf_efficientnet_b4_ns"
    # Extract features from both Penultimate (P4) and Final (P5) blocks
    FEATURE_LEVELS = ["p4", "p5"]
    LSTM_HIDDEN_SIZE = 256
    LSTM_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT_RATE = 0.2
    NUM_CLASSES = 8  # Targets: C1, C2, C3, C4, C5, C6, C7, Patient_Overall

    # --- Training Hyperparameters ---
    EPOCHS = 10
    # Physical batch size is small due to large sequence volume (96 * 384 * 384)
    BATCH_SIZE = 2
    # Gradient accumulation to achieve Effective Batch Size = 16
    GRADIENT_ACCUMULATION_STEPS = 8

    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    MAX_GRAD_NORM = 1000.0

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 3
    EARLY_STOPPING_MIN_DELTA = 1e-4

    # Hardware
    NUM_WORKERS = 12
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Loss Configuration ---
    # Metric: Weighted Multi-Label Log Loss
    # Weights: Patient_Overall is weighted equal to the sum of all vertebrae (7x)
    # Normalized Weights: C_i = 1/14 (~0.071), Overall = 7/14 (0.5)
    # Order: [C1, C2, C3, C4, C5, C6, C7, Patient_Overall]
    CLASS_WEIGHTS = [1 / 14] * 7 + [7 / 14]

    # No positive class weighting to ensure probabilistic calibration
    POS_WEIGHT = 1.0

    @classmethod
    def setup(cls):
        """Creates necessary directories and sets random seeds."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        seed_everything(cls.SEED)


# Initialize setup immediately upon import
Config.setup()
