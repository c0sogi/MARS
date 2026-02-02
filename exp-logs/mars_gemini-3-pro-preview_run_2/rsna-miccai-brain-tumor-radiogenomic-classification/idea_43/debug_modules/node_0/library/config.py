import os
import torch


class Config:
    # --------------------------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------------------------
    SEED = 42

    # --------------------------------------------------------------------------
    # File System & Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    WORKING_DIR = "./working"
    # Specific cache directory for this idea (Idea 43)
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_43")

    # Submission output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Model checkpoint path
    MODEL_PATH = os.path.join(CACHE_DIR, "best_model.pth")

    # Ensure necessary writeable directories exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # --------------------------------------------------------------------------
    # Data Pipeline Hyperparameters
    # --------------------------------------------------------------------------
    IMG_SIZE = 224
    NUM_CHANNELS = 12  # 4 modalities * 3 channels (Rule of 3)

    # Modality Order: Group 1, Group 2, Group 3, Group 4
    # This order is critical for the Asymmetric Grouped EfficientNet architecture
    MODALITIES = ["FLAIR", "T2w", "T1w", "T1wCE"]

    # Modality-Adaptive Stride Configuration
    # FLAIR/T2w (Edema/Fluid) -> Stride 5 for broad context
    # T1w/T1wCE (Texture/Core) -> Stride 2 for fine detail
    STRIDES = {"FLAIR": 5, "T2w": 5, "T1w": 2, "T1wCE": 2}

    # ROI Selection
    ANCHOR_MODALITY = "FLAIR"
    ANCHOR_RANGE = (0.15, 0.85)  # Depth percentage to search for max intensity

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2
    PATIENCE = 5  # For Early Stopping

    # --------------------------------------------------------------------------
    # Compute
    # --------------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # --------------------------------------------------------------------------
    # Debugging
    # --------------------------------------------------------------------------
    DEBUG = False
    DEBUG_SIZE = 50  # Number of samples to use if DEBUG is True
