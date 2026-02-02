import os
import torch


class Config:
    """
    Global configuration for the Whale Identification pipeline (Idea 6).
    """

    # -------------------------------------------------------------------------
    # 1. Reproducibility & Compute
    # -------------------------------------------------------------------------
    SEED = 42
    # Use CUDA if available
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Number of data loading workers (12 vCPUs available, safe to use 4-8)
    NUM_WORKERS = 4

    # -------------------------------------------------------------------------
    # 2. Data Paths
    # -------------------------------------------------------------------------
    # Root Input Directory (Read-Only)
    INPUT_ROOT = "./input"

    # Image Directories
    TRAIN_IMG_DIR = os.path.join(INPUT_ROOT, "train")
    TEST_IMG_DIR = os.path.join(INPUT_ROOT, "test")

    # Metadata Files (Pre-generated in ./metadata)
    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # -------------------------------------------------------------------------
    # 3. Output & Working Directories
    # -------------------------------------------------------------------------
    # Working Directory for Idea 6 (Cache, Checkpoints)
    WORKING_DIR = "./working/idea_6"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Model Checkpoint Path
    MODEL_PATH = os.path.join(WORKING_DIR, "model.pth")

    # Submission Directory and File
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # 4. Model Architecture Hyperparameters
    # -------------------------------------------------------------------------
    MODEL_NAME = "efficientnet_b2"
    # Input resolution (High resolution as per Idea 6)
    IMG_SIZE = 640
    # Dimension of the embedding layer (before ArcFace)
    EMBEDDING_SIZE = 512
    # Dropout rate for the projection head
    DROPOUT_RATE = 0.3

    # -------------------------------------------------------------------------
    # 5. Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 16
    NUM_EPOCHS = 30
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-5

    # ArcFace Head Parameters
    ARC_MARGIN = 0.5
    ARC_SCALE = 30.0

    # -------------------------------------------------------------------------
    # 6. Inference & Validation Hyperparameters
    # -------------------------------------------------------------------------
    # Number of neighbors for Query Expansion / k-Reciprocal Re-ranking
    TOP_K_RETRIEVAL = 50

    # -------------------------------------------------------------------------
    # 7. Debugging / Development
    # -------------------------------------------------------------------------
    # If True, dataset loaders should subset data for quick iteration
    DEBUG = False
    # Number of samples to use when DEBUG is True
    DEBUG_SAMPLE_SIZE = 100
