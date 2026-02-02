import os
import torch


class Config:
    # =========================================================================
    # General Settings
    # =========================================================================
    PROJECT_NAME = "BirdSpeciesClassification_ResNet34_SemiSupervised"
    IDEA_NAME = "idea_5"
    SEED = 42

    # =========================================================================
    # Hardware
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Number of subprocesses for data loading

    # =========================================================================
    # Data Paths
    # =========================================================================
    # Read-only input directories
    INPUT_DIR = "./input"
    ESSENTIAL_DATA_DIR = os.path.join(INPUT_DIR, "essential_data")
    SUPPLEMENTAL_DATA_DIR = os.path.join(INPUT_DIR, "supplemental_data")

    # Spectrogram source
    SPECTROGRAM_DIR = os.path.join(SUPPLEMENTAL_DATA_DIR, "spectrograms")

    # Metadata (Pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output / Working Directory
    WORKING_DIR = os.path.join("./working", IDEA_NAME)

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Model and Submission Artifacts
    MODEL_SAVE_PATH_TEACHER = os.path.join(WORKING_DIR, "teacher_model.pth")
    MODEL_SAVE_PATH_STUDENT = os.path.join(WORKING_DIR, "student_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # Caching Paths (for processed tensors/dataframes)
    CACHE_DIR = WORKING_DIR

    # =========================================================================
    # Data Preprocessing & Augmentation
    # =========================================================================
    # Image Dimensions:
    # Height 256 preserves frequency bin resolution (Nyquist/bins).
    # Width 512 captures full 10s context while densifying features.
    IMG_HEIGHT = 256
    IMG_WIDTH = 512
    IMG_CHANNELS = 3  # ResNet expects 3 channels

    # Mixup
    USE_MIXUP = True
    MIXUP_ALPHA = 0.2

    # =========================================================================
    # Model Architecture
    # =========================================================================
    MODEL_ARCH = "resnet34"
    PRETRAINED = True
    NUM_CLASSES = 19
    DROPOUT_RATE = (
        0.0  # Standard ResNet usually doesn't use dropout before head, but can be added
    )

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Stage 1: Teacher (Train on Labeled Data)
    TEACHER_EPOCHS = 15
    TEACHER_PATIENCE = 5  # Early stopping patience

    # Stage 2/3: Student (Train on Labeled + Pseudo-Labeled Data)
    STUDENT_EPOCHS = 20
    STUDENT_PATIENCE = 5

    # Debugging / Quick Run
    # Set to a small integer (e.g., 100) to limit dataset size for testing pipeline
    DEBUG_SAMPLE_SIZE = None

    @classmethod
    def print_config(cls):
        print(f"--- Configuration: {cls.IDEA_NAME} ---")
        print(f"Device: {cls.DEVICE}")
        print(f"Input Resolution: {cls.IMG_HEIGHT}x{cls.IMG_WIDTH}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Mixup Alpha: {cls.MIXUP_ALPHA}")
        print(f"Teacher Epochs: {cls.TEACHER_EPOCHS}")
        print(f"Student Epochs: {cls.STUDENT_EPOCHS}")
        print(f"Working Directory: {cls.WORKING_DIR}")
        print("---------------------------------------")
