import os
import torch


class Config:
    """
    Centralized configuration for the Herbarium 2020 plant species classification task.
    """

    # =========================================================================
    # Directories and Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_1"
    SUBMISSION_DIR = "./submission"

    # Ensure necessary directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata File Paths
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Output Paths
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_CHECKPOINT_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================
    MODEL_NAME = "resnet34"
    NUM_CLASSES = 32093
    PRETRAINED = True
    DROPOUT_RATE = (
        0.0  # ResNet usually doesn't use dropout in the head, but can be added
    )

    # =========================================================================
    # Data Processing
    # =========================================================================
    IMG_SIZE = (224, 224)
    # Standard ImageNet normalization means and stds
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    BATCH_SIZE = 256  # Optimized for A100 40GB and ResNet-34
    EPOCHS = 15

    # Optimizer (AdamW)
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-2

    # Scheduler (Cosine Annealing)
    T_MAX = EPOCHS  # For CosineAnnealingLR
    MIN_LR = 1e-6

    # Early Stopping
    PATIENCE = 3

    # =========================================================================
    # Hardware and System
    # =========================================================================
    NUM_WORKERS = 12
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    PIN_MEMORY = True

    # =========================================================================
    # Debugging and Development
    # =========================================================================
    # Set to a small integer (e.g., 1000) to train/val on a subset for debugging.
    # Set to None to use the full dataset.
    DEBUG_SAMPLE_SIZE = None
