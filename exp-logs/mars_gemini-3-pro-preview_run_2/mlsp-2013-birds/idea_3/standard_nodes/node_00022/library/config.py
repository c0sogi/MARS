import os
import torch


class Config:
    # --- General ---
    PROJECT_NAME = "BirdSpeciesClassification_Idea3"
    SEED = 42
    DEBUG = False  # Set to True to run on a subset of data for testing

    # --- Directories ---
    ROOT_DIR = "."
    INPUT_DIR = os.path.join(ROOT_DIR, "input")
    METADATA_DIR = os.path.join(ROOT_DIR, "metadata")

    # We use filtered spectrograms as per the idea description
    IMAGE_DIR = os.path.join(INPUT_DIR, "supplemental_data", "filtered_spectrograms")

    # Output directories
    WORKING_DIR = os.path.join(ROOT_DIR, "working", "idea_3")
    SUBMISSION_DIR = os.path.join(ROOT_DIR, "submission")

    # Ensure working directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # --- Data Paths ---
    # Note: For K-Fold, we will likely merge train and val CSVs,
    # but we define their locations here.
    TRAIN_CSV_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # --- Data Parameters ---
    NUM_CLASSES = 19
    # EfficientNet-B0 default is 224.
    # Spectrograms are wide (time axis), so we might use a wider aspect ratio
    # if the transform pipeline supports it, but (224, 224) is standard for transfer learning.
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    CHANNELS = 3  # Pretrained models expect 3 channels (we will replicate the mono spectrogram)

    # --- Model Parameters ---
    BACKBONE = "resnet18"
    PRETRAINED = True
    USE_GEM_POOLING = False

    # --- Training Parameters ---
    N_FOLDS = 5
    EPOCHS = 40
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    PATIENCE = 10  # For Early Stopping

    # Scheduler
    T_MAX = 25  # For CosineAnnealingLR
    MIN_LR = 1e-6

    # --- Compute ---
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def print_config(cls):
        print("=" * 30)
        print(f"Configuration: {cls.PROJECT_NAME}")
        print("=" * 30)
        for key, value in cls.__dict__.items():
            if not key.startswith("__") and not callable(value):
                print(f"{key}: {value}")
        print("=" * 30)
