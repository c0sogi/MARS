import os
import torch


class Config:
    """
    Centralized configuration for the Translation-Invariant SegFormer (MiT-B2)
    with Discrete Multi-View Training.
    """

    # --- General Configuration ---
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Use available CPUs for data loading, capping at a reasonable number to avoid overhead
    NUM_WORKERS = min(os.cpu_count(), 12)

    # --- File Paths ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Specific working directory for this idea's cached data and checkpoints
    WORKING_DIR = "./working/idea_20"
    # Output path for the final submission file
    SUBMISSION_PATH = "./submission.csv"

    # --- Data Processing ---
    TILE_SIZE = 512
    STRIDE = 512

    # --- Discrete Multi-View Protocol ---
    # We define a slab thickness of 12 slices.
    # We define three discrete starting positions for the Z-axis views.
    SLAB_DEPTH = 12
    VIEW_A_START = 16  # High view (lower indices)
    VIEW_B_START = 20  # Center view
    VIEW_C_START = 24  # Low view (higher indices)

    # Input channels for the model (using ImageNet pre-trained weights)
    IN_CHANNELS = 3

    # --- Model Architecture ---
    MODEL_NAME = "nvidia/mit-b2"

    # --- Training Hyperparameters ---
    # Batch size set to 8 as per Micro-Dataset Optimization Protocol
    BATCH_SIZE = 8
    # Conservative learning rate to prevent divergence
    LEARNING_RATE = 6e-5
    # Number of training epochs
    EPOCHS = 15

    # --- Optimization & Regularization ---
    WEIGHT_DECAY = 0.01
    PATIENCE = 5  # Early stopping patience

    # --- Inference ---
    # Threshold for converting probability maps to binary masks
    THRESHOLD = 0.5

    @classmethod
    def setup(cls):
        """
        Initialize the environment by creating necessary directories.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)


# Execute setup on import to ensure directories exist
Config.setup()
