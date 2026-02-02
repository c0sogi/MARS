import os
import torch


class Config:
    # ==========================================
    # Path Configuration
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Specific directories
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")

    # Metadata files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Output paths
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")

    # Ensure directories exist
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    os.makedirs(MODEL_CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(os.path.join(WORKING_DIR, "idea_1"), exist_ok=True)

    # ==========================================
    # Data Configuration
    # ==========================================
    SEED = 42
    IMG_SIZE = 224
    NUM_WORKERS = 4  # optimized for the available vCPUs

    # Class Labels (Sorted Alphabetically as per Data Analysis)
    CLASS_LABELS = [
        "complex",
        "frog_eye_leaf_spot",
        "healthy",
        "powdery_mildew",
        "rust",
        "scab",
    ]
    NUM_CLASSES = len(CLASS_LABELS)

    # ==========================================
    # Model & Training Configuration
    # ==========================================
    MODEL_NAME = "resnet18"
    PRETRAINED = True

    # Hyperparameters
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    EPOCHS = 15

    # Inference Threshold
    THRESHOLD = 0.5

    # Compute
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Debugging / Development
    # Set to a small integer (e.g., 100) to debug pipeline with a subset of data
    # Set to None for full training
    DEBUG_SAMPLE_SIZE = None
