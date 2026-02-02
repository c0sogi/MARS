import os
import torch


class Config:
    # ==========================================
    # System Settings
    # ==========================================
    SEED = 42
    # Check for GPU availability
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Number of data loading workers (adjusted for 12 vCPUs)
    NUM_WORKERS = 4

    # ==========================================
    # File Paths
    # ==========================================
    # Root directory containing the 'train_images' and 'test_images' folders
    INPUT_ROOT = "./input"

    # Directory containing the generated metadata CSVs
    METADATA_DIR = "./metadata"
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output directories for model checkpoints and intermediate files
    WORKING_DIR = "./working/idea_1"
    SUBMISSION_DIR = "./submission"

    # Ensure output directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Specific output file paths
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "resnet18_best.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Configuration
    # ==========================================
    # Input resolution for ResNet-18
    IMG_SIZE = 224
    # RGB Channels
    IN_CHANNELS = 3
    # Total number of classes (0-22)
    NUM_CLASSES = 23

    # ImageNet Normalization Constants
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # ==========================================
    # Model Configuration
    # ==========================================
    MODEL_NAME = "resnet18"
    PRETRAINED = True

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 64
    EPOCHS = 10
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4

    # Early Stopping settings
    EARLY_STOPPING_PATIENCE = 3
    EARLY_STOPPING_MIN_DELTA = 0.001

    # ==========================================
    # Debug / Development
    # ==========================================
    # If True, trains on a small subset of data for quick verification
    DEBUG = False
    DEBUG_SUBSET_SIZE = 2000
