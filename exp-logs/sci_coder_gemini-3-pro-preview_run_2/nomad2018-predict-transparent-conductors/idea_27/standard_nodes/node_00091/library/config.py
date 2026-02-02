import os
import torch


class Config:
    """
    Configuration class for the Stoichiometry-Preserving Receiver-Aware Crystal Graph Network (SP-RA-CGN).
    Centralizes all hyperparameters, file paths, and system settings.
    """

    # ==========================================
    # Path Configuration
    # ==========================================
    # Base directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_27"
    SUBMISSION_DIR = "./submission"

    # Sub-directories for artifacts
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")

    # Input Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Cached Data Files (Processed Graphs)
    # Using .npz for efficient storage of graph data
    TRAIN_GRAPHS_PATH = os.path.join(CACHE_DIR, "train_graphs.npz")
    VAL_GRAPHS_PATH = os.path.join(CACHE_DIR, "val_graphs.npz")
    TEST_GRAPHS_PATH = os.path.join(CACHE_DIR, "test_graphs.npz")

    # Scaler State
    TARGET_SCALER_PATH = os.path.join(CACHE_DIR, "target_scaler.npz")

    # Model Checkpoint
    MODEL_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")

    # Final Submission
    SUBMISSION_FILE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Processing Hyperparameters
    # ==========================================
    # Graph Construction
    CUTOFF_RADIUS = 5.0  # Angstroms, strict cutoff for local interactions
    MAX_NEIGHBORS = 12  # Maximum number of neighbors to consider per node (Cite solution_lesson_node_00078)

    # Edge Features (Gaussian RBF)
    RBF_BINS = 60  # Number of Gaussian basis functions
    RBF_LOWER = 0.0  # Lower bound for RBF (Angstroms)
    RBF_UPPER = 5.0  # Upper bound for RBF (Angstroms)

    # ==========================================
    # Model Architecture Hyperparameters
    # ==========================================
    # Embedding Dimensions
    ATOM_EMBEDDING_DIM = 128  # Dimension of initial atom embeddings (h^0)
    HIDDEN_DIM = 128  # Dimension of hidden layers in interaction blocks
    EDGE_EMBEDDING_DIM = 128  # Dimension to project edge RBF features into

    # Network Depth
    NUM_INTERACTION_BLOCKS = 4  # Number of receiver-aware interaction layers

    # Regularization & Stability
    DROPOUT_RATE = 0.1  # Dropout probability
    RESIDUAL_INIT = 0.0  # Initial value for learnable residual scalar epsilon

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42  # Random seed for reproducibility
    BATCH_SIZE = 48  # Batch size for training and evaluation
    NUM_EPOCHS = 150  # Maximum number of training epochs
    LEARNING_RATE = 1e-3  # Initial learning rate
    WEIGHT_DECAY = 1e-4  # Weight decay for AdamW optimizer

    # Scheduler (ReduceLROnPlateau)
    SCHEDULER_FACTOR = 0.5  # Factor to reduce LR by
    SCHEDULER_PATIENCE = 10  # Epochs with no improvement before reducing LR
    SCHEDULER_MIN_LR = 1e-6  # Minimum learning rate

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 20  # Epochs with no improvement before stopping

    # ==========================================
    # System Settings
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Number of dataloader workers

    @classmethod
    def setup(cls):
        """
        Creates necessary working directories if they do not exist.
        """
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        print(f"Directories initialized: {cls.WORKING_DIR}")
        print(f"Device selected: {cls.DEVICE}")
