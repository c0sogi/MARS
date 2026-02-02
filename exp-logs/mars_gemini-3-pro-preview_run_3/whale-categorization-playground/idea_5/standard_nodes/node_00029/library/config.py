import os
import torch
import random
import numpy as np


class Config:
    # -------------------------------------------------------------------------
    # Reproducibility & Environment
    # -------------------------------------------------------------------------
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4  # Number of data loading workers

    @staticmethod
    def setup_reproducibility(seed=42):
        """Sets the seed for all random number generators to ensure reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)

    # -------------------------------------------------------------------------
    # File Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata CSVs (Pre-generated)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Working Directory for Idea 5
    WORKING_DIR = "./working/idea_5"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Cache Directory for processed data
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Output Paths
    MODEL_PATH = os.path.join(WORKING_DIR, "efficientnet_b2_arcface.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Data Configuration
    # -------------------------------------------------------------------------
    IMAGE_SIZE = 448  # Native resolution for EfficientNet-B2

    # Debugging / Development
    DEBUG = False  # Set to True to run on a small subset
    DEBUG_SAMPLE_SIZE = 100  # Number of samples to use in debug mode

    # -------------------------------------------------------------------------
    # Model Architecture
    # -------------------------------------------------------------------------
    BACKBONE = "efficientnet_b2"
    EMBEDDING_DIM = 512
    PRETRAINED = True
    DROPOUT = 0.3

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 16  # Small batch size for frequent updates
    LEARNING_RATE = (
        3e-4  # Conservative LR for stability (Cite solution_lesson_node_00028)
    )
    WEIGHT_DECAY = 1e-4
    EPOCHS = 30  # Extended schedule for margin loss (Cite solution_lesson_node_00019)

    # ArcFace Loss Parameters
    CF_S = 30.0  # Scale factor
    CF_M = 0.5  # Margin

    # Learning Rate Scheduler
    SCHEDULER_PATIENCE = 2
    SCHEDULER_FACTOR = 0.1
    MIN_LR = 1e-6

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 5

    # -------------------------------------------------------------------------
    # Inference & Post-Processing
    # -------------------------------------------------------------------------
    KNN_K = 100  # Number of neighbors for retrieval
    NEW_WHALE_THRESH = 0.45  # Threshold for assigning 'new_whale'

    # Re-ranking Parameters (k-Reciprocal Encoding)
    RERANK_K1 = 20
    RERANK_K2 = 6
    RERANK_LAMBDA = 0.3
