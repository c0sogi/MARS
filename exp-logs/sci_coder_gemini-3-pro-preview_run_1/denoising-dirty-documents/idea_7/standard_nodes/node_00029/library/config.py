import os
import torch


class Config:
    """
    Centralized configuration for the Denoising Task.
    Implements settings for Resolution-Scaled 4-Level U-Net and Seed-Averaging Ensemble.
    """

    # -------------------------------------------------------------------------
    # Paths & Directories
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Specific working directory for this experiment (Idea 7)
    WORKING_DIR = "./working/idea_7"

    # Submission output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Metadata File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Ensure writable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Data Configuration
    # -------------------------------------------------------------------------
    # Patch size set to 160x160 to maximize data diversity via random crops (Cite solution_lesson_node_00027)
    PATCH_SIZE = 160

    BATCH_SIZE = 16
    NUM_WORKERS = 4  # Leveraging available vCPUs

    # -------------------------------------------------------------------------
    # Model Configuration
    # -------------------------------------------------------------------------
    # Lightweight 3-Level U-Net Architecture (Cite solution_lesson_node_00018, solution_lesson_node_00026)
    # Encoder Filters: 32 -> 64 -> 128 -> 256 (Bottleneck)
    MODEL_DEPTH = 3
    ENCODER_FILTERS = [32, 64, 128, 256]

    # -------------------------------------------------------------------------
    # Training Configuration
    # -------------------------------------------------------------------------
    # Training for full convergence as per strategy
    EPOCHS = 1000

    # High learning rate with Adam
    LEARNING_RATE = 1e-3

    # Cosine Annealing Scheduler settings
    T_MAX = 1000  # Matches EPOCHS
    ETA_MIN = 0.0

    # Ensemble Strategy: 5 independent models
    SEEDS = [42, 43, 44, 45, 46]

    # -------------------------------------------------------------------------
    # Inference Configuration
    # -------------------------------------------------------------------------
    # Test-Time Augmentation (TTA)
    # 8 Views: Original, Rot90, Rot180, Rot270 + Horizontal Flips of each
    TTA_VIEWS = 8

    # -------------------------------------------------------------------------
    # Hardware & Debugging
    # -------------------------------------------------------------------------
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Debugging / Development flags
    # Set DEBUG = True to train on a small subset for pipeline verification
    DEBUG = False
    MAX_DEBUG_SAMPLES = 100
