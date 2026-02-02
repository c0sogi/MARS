import os
import torch


class Config:
    # ==========================================
    # Directories & Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_10"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Data Source
    SPECTROGRAM_DIR = os.path.join(INPUT_DIR, "supplemental_data", "spectrograms")

    # Output Paths
    SUBMISSION_PATH = "./submission/submission.csv"

    # ==========================================
    # Data Hyperparameters
    # ==========================================
    # High-Fidelity Resolution-Aligned Dimensions
    IMG_HEIGHT = 256
    IMG_WIDTH = 640

    NUM_CLASSES = 19
    BATCH_SIZE = 32
    NUM_WORKERS = 4

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    BACKBONE = "resnet34"
    PRETRAINED = True

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    LEARNING_RATE = 3e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Augmentation
    MIXUP_ALPHA = 0.2

    # ==========================================
    # Stage 1: Teacher Ensemble Config
    # ==========================================
    NUM_TEACHERS = 3
    TEACHER_EPOCHS = 40
    # SWA starts at 75% of training
    TEACHER_SWA_START_RATIO = 0.75
    TEACHER_SWA_LR = 1e-4

    # ==========================================
    # Stage 3: Student Config
    # ==========================================
    STUDENT_EPOCHS = 50
    # SWA starts at 70% of training
    STUDENT_SWA_START_RATIO = 0.70
    STUDENT_SWA_LR = 1e-4

    # ==========================================
    # Debugging
    # ==========================================
    DEBUG = False
    DEBUG_SUBSET_SIZE = 50  # Number of samples to use when DEBUG is True

    @classmethod
    def get_swa_start_epoch(cls, total_epochs, ratio):
        """Calculates the epoch at which SWA should start."""
        return int(total_epochs * ratio)
