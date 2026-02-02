import os
import torch


class Config:
    """
    Configuration class for the Brain Tumor Radiogenomic Classification task.
    Implements Idea 9: Multi-Slice Data Expansion with Brain-Centric Geometric Sampling.
    """

    # ==========================================
    # Global Constants & Reproducibility
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # Directory & Path Setup
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_9/"

    # Ensure working directory exists for caching and outputs
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Submission Output Path
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Data Preprocessing & Sampling (Idea 9)
    # ==========================================
    IMAGE_SIZE = 224
    NUM_CHANNELS = 3  # Channel 1: FLAIR, Channel 2: T1wCE, Channel 3: T2w

    # Deterministic Data Expansion:
    # Instead of a single middle slice, we extract 3 slices at specific relative depths
    # within the brain bounding box.
    SAMPLING_DEPTHS = [0.45, 0.50, 0.55]

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    MODEL_NAME = "efficientnet_b0"
    PRETRAINED = True
    NUM_CLASSES = 1  # Binary classification (MGMT promoter methylation)

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 32
    NUM_WORKERS = 4  # Optimized for 12 vCPUs

    # Optimization
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2  # Strong regularization for small dataset

    # Training Loop
    EPOCHS = 15
    N_FOLDS = 5
    EARLY_STOPPING_PATIENCE = 5

    # ==========================================
    # Caching Configuration
    # ==========================================
    # Files to store processed numpy arrays/parquets to avoid re-reading DICOMs
    CACHE_TRAIN_PATH = os.path.join(WORKING_DIR, "cache_train_data.parquet")
    CACHE_VAL_PATH = os.path.join(WORKING_DIR, "cache_val_data.parquet")
    CACHE_TEST_PATH = os.path.join(WORKING_DIR, "cache_test_data.parquet")

    # ==========================================
    # Debugging / Development
    # ==========================================
    # If set to an integer (e.g., 20), only use that many subjects for training/testing.
    # Useful for checking pipeline mechanics without full runtime.
    # Set to None for the full competition run.
    DEBUG_SAMPLE_SIZE = None
