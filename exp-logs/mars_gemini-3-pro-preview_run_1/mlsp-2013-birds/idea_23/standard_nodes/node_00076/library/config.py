import os
import torch


class Config:
    """
    Global configuration for the Sharpness-Aware High-Fidelity SWA-Distillation pipeline.
    """

    # ==========================================
    # Reproducibility
    # ==========================================
    SEED = 42

    # ==========================================
    # File Paths & Directories
    # ==========================================
    # Input Data
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    SPECTROGRAM_DIR = os.path.join(INPUT_DIR, "supplemental_data", "spectrograms")

    # Metadata CSVs
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Working Directory (Checkpoints, Cache, Logs)
    WORKING_DIR = "./working/idea_23"
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # Pseudo-labels
    PSEUDO_LABEL_PATH = os.path.join(WORKING_DIR, "pseudo_labels.parquet")

    # Submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Hyperparameters
    # ==========================================
    NUM_CLASSES = 19

    # High-Fidelity Resolution
    IMG_HEIGHT = 256
    IMG_WIDTH = 640
    CHANNELS = 3  # Replicating grayscale to RGB

    # ImageNet Normalization
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # Augmentation
    MIXUP_ALPHA = 0.2

    # Data Loading
    NUM_WORKERS = 4
    PIN_MEMORY = True

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    MODEL_NAME = "resnet34"
    PRETRAINED = True
    DROPOUT = 0.0  # Disabled in favor of SAM

    # ==========================================
    # Optimization Hyperparameters
    # ==========================================
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Sharpness-Aware Minimization (SAM)
    USE_SAM = True
    SAM_RHO = 0.05

    # ==========================================
    # Training Schedules
    # ==========================================
    # Stage 1: Teacher Ensemble
    NUM_TEACHERS = 3
    EPOCHS_TEACHER = 50
    # Activate SWA in the final 25% of epochs
    SWA_START_EPOCH_TEACHER = int(EPOCHS_TEACHER * 0.75)  # 37

    # Stage 3: Student Training
    EPOCHS_STUDENT = 50
    # Activate SWA in the final 30% of epochs
    SWA_START_EPOCH_STUDENT = int(EPOCHS_STUDENT * 0.70)  # 35

    # ==========================================
    # Compute
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """
        Creates necessary output directories for the experiment.
        """
        dirs = [cls.WORKING_DIR, cls.CHECKPOINT_DIR, cls.CACHE_DIR, cls.SUBMISSION_DIR]
        for d in dirs:
            os.makedirs(d, exist_ok=True)

        print(f"Directories initialized at {cls.WORKING_DIR}")

    @classmethod
    def print_summary(cls):
        """
        Prints a summary of the configuration.
        """
        print("=" * 40)
        print("CONFIG SUMMARY")
        print("=" * 40)
        print(f"Device: {cls.DEVICE}")
        print(f"Model: {cls.MODEL_NAME} (Pretrained={cls.PRETRAINED})")
        print(f"Resolution: {cls.IMG_HEIGHT}x{cls.IMG_WIDTH}")
        print(f"Optimizer: SAM (rho={cls.SAM_RHO})")
        print(f"Mixup Alpha: {cls.MIXUP_ALPHA}")
        print(f"Teachers: {cls.NUM_TEACHERS} models, {cls.EPOCHS_TEACHER} epochs each")
        print(f"Student: 1 model, {cls.EPOCHS_STUDENT} epochs")
        print("=" * 40)
