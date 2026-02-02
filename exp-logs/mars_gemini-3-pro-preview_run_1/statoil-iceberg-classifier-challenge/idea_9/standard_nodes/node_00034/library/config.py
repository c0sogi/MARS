import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Config:
    # ==========================================
    # General Configuration
    # ==========================================
    PROJECT_NAME = "Iceberg_ResNet18_Distillation"
    SEED = 42
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 100  # Number of samples to use when DEBUG is True

    # Compute
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # File Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_9"

    # Raw Data
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata (Pre-generated)
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Artifacts & Outputs
    TEACHER_CHECKPOINT_DIR = os.path.join(WORKING_DIR, "teacher_checkpoints")
    STUDENT_CHECKPOINT_DIR = os.path.join(WORKING_DIR, "student_checkpoints")
    OOF_PREDICTIONS_PATH = os.path.join(WORKING_DIR, "oof_predictions.npy")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # ==========================================
    # Data Processing
    # ==========================================
    IMAGE_SIZE = 224
    NUM_CHANNELS = 3  # Band 1, Band 2, Mean

    # Augmentation
    ROTATION_DEGREES = 20  # Continuous rotation range (+/-)

    # ==========================================
    # Model Architecture
    # ==========================================
    MODEL_ARCH = "resnet18"
    PRETRAINED = True
    NUM_CLASSES = 1
    DROPOUT_RATE = 0.5

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    N_FOLDS = 5
    BATCH_SIZE = 32
    NUM_EPOCHS = 30
    EARLY_STOPPING_PATIENCE = 7

    # Optimization
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.01

    # Scheduler (ReduceLROnPlateau)
    SCHEDULER_FACTOR = 0.1
    SCHEDULER_PATIENCE = 3

    # ==========================================
    # Loss & Distillation
    # ==========================================
    LABEL_SMOOTHING = 0.05

    # Distillation Loss: Loss = Alpha * BCE(True, Pred) + (1 - Alpha) * KL(Soft, Pred)
    DISTILLATION_ALPHA = 0.5

    @classmethod
    def create_directories(cls):
        """
        Creates necessary working directories for checkpoints and outputs.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.TEACHER_CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.STUDENT_CHECKPOINT_DIR, exist_ok=True)
        print(f"Directories created at {cls.WORKING_DIR}")

    @classmethod
    def print_config(cls):
        """
        Prints the current configuration settings.
        """
        print("=" * 40)
        print(f"CONFIGURATION: {cls.PROJECT_NAME}")
        print("=" * 40)
        for k, v in cls.__dict__.items():
            if not k.startswith("__") and not callable(v):
                print(f"{k:<25}: {v}")
        print("=" * 40)
