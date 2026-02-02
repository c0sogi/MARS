import os
import torch


class Config:
    # -------------------------------------------------------------------------
    # Directories & Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test")

    METADATA_DIR = "./metadata"
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Working directory for checkpoints, cached data, and logs
    WORKING_DIR = "./working/idea_33"

    # Submission directory
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Data Configuration
    # -------------------------------------------------------------------------
    IMAGE_SIZE = 32
    NUM_CLASSES = 1

    # -------------------------------------------------------------------------
    # Model Configuration (Custom RepVGG)
    # -------------------------------------------------------------------------
    # Standard Channel Configuration for the 3 stages (Cite solution_lesson_node_00019)
    CHANNELS = [32, 64, 128]
    # Cardinality = 1 for standard convolution (RepVGG style) (Cite solution_lesson_node_00103)
    CARDINALITY = 1

    # -------------------------------------------------------------------------
    # Training Configuration
    # -------------------------------------------------------------------------
    # Homogeneous Seed Averaging: 5 independent instances
    SEEDS = [0, 1, 2, 3, 4]

    EPOCHS = 15
    BATCH_SIZE = 128

    # Optimizer (AdamW) & Scheduler (Cosine Annealing)
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2
    T_MAX = 20
    ETA_MIN = 1e-6

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 5

    # -------------------------------------------------------------------------
    # Hardware & Utilities
    # -------------------------------------------------------------------------
    NUM_WORKERS = 2
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """Creates necessary output directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Execute setup on import to ensure directories exist
Config.setup()
