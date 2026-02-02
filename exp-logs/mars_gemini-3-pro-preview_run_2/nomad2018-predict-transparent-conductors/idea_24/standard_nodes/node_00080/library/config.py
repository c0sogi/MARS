import os
import torch


class Config:
    # -------------------------------------------------------------------------
    # File Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for caching processed data and checkpoints
    # Using a specific subdirectory for this idea to avoid conflicts
    WORKING_DIR = "./working/idea_24"

    # Submission directory
    SUBMISSION_DIR = "./submission"

    # Metadata file paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Cache file paths for processed graphs (numpy .npz format)
    TRAIN_GRAPHS_PATH = os.path.join(WORKING_DIR, "train_graphs.npz")
    VAL_GRAPHS_PATH = os.path.join(WORKING_DIR, "val_graphs.npz")
    TEST_GRAPHS_PATH = os.path.join(WORKING_DIR, "test_graphs.npz")

    # Scaler path for target normalization
    TARGET_SCALER_PATH = os.path.join(WORKING_DIR, "target_scaler.npz")

    # Model checkpoint path
    CHECKPOINT_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Final submission file path
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------------------
    SEED = 42

    # -------------------------------------------------------------------------
    # Data Processing Hyperparameters
    # -------------------------------------------------------------------------
    # Graph Topology: k-Nearest Neighbors
    K_NEIGHBORS = 12

    # Geometric Features (RBF)
    RBF_BINS = 60
    RBF_CUTOFF = 8.0  # Angstroms (upper bound for RBF centers)

    # Debugging: Set to an integer (e.g., 100) to train/test on a small subset
    # Set to None for full dataset training
    DEBUG_SAMPLE_SIZE = None

    # -------------------------------------------------------------------------
    # Model Architecture Hyperparameters (k-RA-CGN)
    # -------------------------------------------------------------------------
    HIDDEN_DIM = 128
    NUM_BLOCKS = 4
    DROPOUT = 0.1

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 48
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 150
    PATIENCE = 15  # Early stopping patience

    # Compute device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """
        Creates necessary output directories if they do not exist.
        Should be called at the start of the pipeline.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
