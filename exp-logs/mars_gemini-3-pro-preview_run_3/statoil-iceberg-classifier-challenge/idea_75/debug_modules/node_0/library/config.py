import os
import torch


class Config:
    """
    Configuration for the Capacity-Constrained Tri-Statistic Isomorphic CNN (CCTI-CNN) experiment.
    """

    # -------------------------------------------------------------------------
    # Experiment Identity & Reproducibility
    # -------------------------------------------------------------------------
    EXPERIMENT_NAME = "idea_75"
    SEED = 42

    # -------------------------------------------------------------------------
    # Compute Environment
    # -------------------------------------------------------------------------
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------------------------------------------------------
    # File Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for this specific idea
    WORKING_DIR = os.path.join("./working", EXPERIMENT_NAME)

    # Sub-directories for artifacts
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")

    # Final output path
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Raw Data Files (referenced via metadata, but paths kept here for safety)
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")

    # Metadata Files
    TRAIN_META_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_CSV = os.path.join(METADATA_DIR, "test.csv")

    # -------------------------------------------------------------------------
    # Data Specifications
    # -------------------------------------------------------------------------
    IMAGE_SIZE = 75
    IN_CHANNELS = 3  # HH, HV, Avg((HH+HV)/2)
    NUM_CLASSES = 1  # Binary classification

    # -------------------------------------------------------------------------
    # Model Architecture Hyperparameters (CCTI-CNN)
    # -------------------------------------------------------------------------
    # Backbone: 4-Stage Plain CNN
    # Width strategy: Expand early, then cap to prevent explosion
    BACKBONE_CHANNELS = [64, 128, 128, 128]

    # Attention: Hybrid SE Module
    SE_REDUCTION = 16

    # Tri-Statistic Isomorphic Readout
    # We extract features from Stage 3 and Stage 4 (0-indexed: 2 and 3)
    EXTRACT_INDICES = [2, 3]

    # Projection: Decoupled 1x1 convs to preserve texture width
    PROJECTION_DIM = 64

    # Capacity-Constrained Head
    # Input dim will be: PROJECTION_DIM * 3 (stats) * 2 (stages) + 1 (angle) = 385
    # We constrain hidden dim to 128 to manage parameter budget (~49k params)
    HEAD_HIDDEN_DIM = 128
    DROPOUT_RATE = 0.5

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    NUM_FOLDS = 5
    BATCH_SIZE = 32

    # Optimizer: AdamW with constant LR (no scheduler)
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4  # L2 Regularization

    # Loop controls
    NUM_EPOCHS = 75
    PATIENCE = 12  # Early stopping patience

    # -------------------------------------------------------------------------
    # Debugging / Development
    # -------------------------------------------------------------------------
    # If True, runs on a small subset of data for pipeline verification
    DEBUG = False
    DEBUG_SUBSET_SIZE = 100

    @classmethod
    def create_directories(cls):
        """
        Creates the necessary directory structure for the experiment.
        Safe to call multiple times.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        print(f"Directories initialized at {cls.WORKING_DIR}")
