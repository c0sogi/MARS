import os
import torch


class Config:
    """
    Configuration class for the Artwork Attribute Labeling task.
    Centralizes hyperparameters, file paths, and model settings.
    """

    # --- Path Configuration ---
    INPUT_DIR = "./input"
    TRAIN_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DIR = os.path.join(INPUT_DIR, "test")

    METADATA_DIR = "./metadata"
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
    LABELS_PATH = os.path.join(INPUT_DIR, "labels.csv")

    # Working directory for checkpoints and cache
    WORKING_DIR = "./working/idea_2"

    # Submission directory
    SUBMISSION_DIR = "./submission"

    # Output file paths
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --- Model Configuration ---
    # Using ConvNeXt-Tiny as the backbone for better feature extraction
    MODEL_NAME = "convnext_tiny"
    PRETRAINED = True
    # Increased input resolution for fine-grained details
    IMAGE_SIZE = 384
    NUM_CLASSES = 3474

    # --- Training Configuration ---
    SEED = 42
    # Batch size adjusted for 384x384 resolution on A100 GPU
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 1e-2
    NUM_WORKERS = 8
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Mixed Precision Training
    USE_FP16 = True

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 3

    # --- Asymmetric Loss Configuration ---
    # Parameters to down-weight easy negatives and focus on hard samples
    ASL_GAMMA_NEG = 4.0
    ASL_GAMMA_POS = 0.0
    ASL_CLIP = 0.05

    # --- Debugging Configuration ---
    # Set DEBUG to True to run on a small subset of data
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 2000

    @classmethod
    def setup(cls):
        """
        Creates necessary output directories if they do not exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
