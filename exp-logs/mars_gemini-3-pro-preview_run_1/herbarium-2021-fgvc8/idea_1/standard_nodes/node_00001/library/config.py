import os
import torch


class Config:
    # ==== General Settings ====
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset of data for debugging
    DEBUG_SAMPLE_SIZE = 1000  # Number of samples to use in debug mode

    # ==== Paths ====
    # Input directories (Read-Only)
    INPUT_DIR = "./input"
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train", "images")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test", "images")

    # Metadata paths (Generated previously)
    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Output directories
    WORKING_DIR = "./working"
    IDEA_DIR = os.path.join(WORKING_DIR, "idea_1")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure output directories exist
    os.makedirs(IDEA_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==== Data Parameters ====
    NUM_CLASSES = 64500  # Based on EDA: 64500 unique categories in Train
    IMAGE_SIZE = 224  # Resizing to 224x224 for ResNet-18
    NUM_WORKERS = 12  # 12 vCPUs available

    # ==== Model Parameters ====
    MODEL_NAME = "resnet18"
    PRETRAINED = True
    DROPOUT = 0.0  # ResNet usually doesn't use dropout in backbone, but head might

    # ==== Training Hyperparameters ====
    BATCH_SIZE = 256  # A100 40GB can handle large batches for ResNet18
    NUM_EPOCHS = 15  # One-Cycle policy converges fast
    LEARNING_RATE = 1e-3  # Max LR for One-Cycle
    WEIGHT_DECAY = 1e-4  # Standard weight decay
    LABEL_SMOOTHING = 0.1

    # ==== Compute ====
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    PIN_MEMORY = True

    # ==== Early Stopping ====
    PATIENCE = 3  # Stop if validation loss doesn't improve for 3 epochs

    @classmethod
    def print_config(cls):
        """Prints the current configuration."""
        print("=" * 30)
        print("CONFIG")
        print("=" * 30)
        for key, value in cls.__dict__.items():
            if not key.startswith("__") and not callable(value):
                print(f"{key}: {value}")
        print("=" * 30)
