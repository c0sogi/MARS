import os
import torch


class Config:
    # ==========================
    # Paths
    # ==========================
    INPUT_DIR = "./input"
    TRAIN_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DIR = os.path.join(INPUT_DIR, "test")

    METADATA_DIR = "./metadata"
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    WORKING_DIR = "./working/idea_3"
    CHECKPOINT_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = "./submission/submission.csv"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    # ==========================
    # General Settings
    # ==========================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Number of dataloader workers

    # ==========================
    # Data & Preprocessing
    # ==========================
    IMG_SIZE = (256, 256)  # Height, Width
    IN_CHANNELS = 3  # 2.5D: slice i-1, i, i+1
    NUM_CLASSES = 3  # Large Bowel, Small Bowel, Stomach

    # Sampling strategy for training
    # Fraction of negative samples (background only) to keep in training
    NEGATIVE_SAMPLE_RATIO = 0.2

    # ==========================
    # Model Architecture
    # ==========================
    BACKBONE = "mobilenet_v2"
    OUTPUT_STRIDE = 16  # For DeepLabV3+ (controls dilation)

    # ==========================
    # Training Hyperparameters
    # ==========================
    BATCH_SIZE = 32
    EPOCHS = 15
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4  # For AdamW

    # Scheduler (StepLR)
    SCHEDULER_STEP_SIZE = 5
    SCHEDULER_GAMMA = 0.1

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 5

    # ==========================
    # Post-Processing
    # ==========================
    CONFIDENCE_THRESHOLD = 0.5

    # Minimum volume (in voxels) for a 3D connected component to be kept.
    # This removes small noise blobs in the 3D reconstruction.
    MIN_VOLUME_THRESHOLD = 100
