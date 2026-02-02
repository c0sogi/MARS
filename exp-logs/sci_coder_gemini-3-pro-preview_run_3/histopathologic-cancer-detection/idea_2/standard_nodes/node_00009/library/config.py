import os
import torch


class Config:
    """
    Global configuration for the Digital Pathology Tumor Detection task.
    Implements settings for a Heterogeneous Ensemble Pipeline.
    """

    # --- Reproducibility ---
    SEED = 42

    # --- File System Paths ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_3"
    SUBMISSION_DIR = "./submission"

    # Ensure output directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata Paths
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --- Data Processing ---
    # Original image dimensions provided in dataset
    ORIGINAL_SIZE = 96

    # Crop size for training and inference
    # 64x64 Center Crop focuses on the 32x32 ROI with 16px context buffer
    CROP_SIZE = 64

    # --- Model Architectures ---
    # List of backbones for the heterogeneous ensemble
    # Names correspond to 'timm' library identifiers
    MODELS = ["resnet18"]

    # --- Training Hyperparameters ---
    # Optimized for A100-40GB
    BATCH_SIZE = 256
    NUM_EPOCHS = 20
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-2  # Standard for AdamW

    # Compute resources
    NUM_WORKERS = 8  # Utilizing available vCPUs
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Validation & Inference ---
    # Enable Test Time Augmentation (Horizontal/Vertical Flips)
    USE_TTA = True

    # Debugging flag (set to True to run on a small subset of data)
    DEBUG = False
