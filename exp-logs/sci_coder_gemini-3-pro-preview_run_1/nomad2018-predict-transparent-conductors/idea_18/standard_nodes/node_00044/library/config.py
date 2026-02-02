import os
import torch


class Config:
    """
    Configuration for the Chemically-Resolved Neighborhood Deep Sets (CRN-DS) pipeline.
    """

    # ==========================================
    # Directories and Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_18"
    SUBMISSION_DIR = "./submission"

    # Metadata Files
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Cache Files (using .npz for numpy arrays as requested)
    TRAIN_CACHE_PATH = os.path.join(WORKING_DIR, "train_data.npz")
    VAL_CACHE_PATH = os.path.join(WORKING_DIR, "val_data.npz")
    TEST_CACHE_PATH = os.path.join(WORKING_DIR, "test_data.npz")
    SCALERS_CACHE_PATH = os.path.join(WORKING_DIR, "scalers.npz")

    # Model Checkpoint
    MODEL_CHECKPOINT_PATH = os.path.join(WORKING_DIR, "best_model.pt")

    # Submission Output
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Processing Constants
    # ==========================================
    # Atomic types to consider for one-hot encoding and neighbor distances
    ATOM_TYPES = ["Al", "Ga", "In", "O"]

    # Value to use for distance when a specific chemical neighbor is not present
    # This acts as an "infinite" distance for the network
    MAX_NEIGHBOR_DISTANCE = 10.0

    # Feature Dimensions
    # Atomic Stream Input:
    #   4 (One-Hot Identity) +
    #   3 (Centered Coordinates x,y,z) +
    #   4 (Chemically-Resolved Neighbor Distances)
    #   = 11
    ATOMIC_FEATURE_DIM = 11

    # Global Stream Input:
    #   3 (Lattice Lengths) +
    #   3 (Lattice Angles) +
    #   1 (Unit Cell Volume) +
    #   1 (Atomic Density) +
    #   3 (Stoichiometry Al, Ga, In) +
    #   1 (Total Atoms)
    #   = 12
    GLOBAL_FEATURE_DIM = 12

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    # Atomic Stream Encoder (Wide MLP)
    ATOMIC_HIDDEN_DIM = 512
    ATOMIC_LATENT_DIM = 128  # Dimension after pooling

    # Global Stream Encoder
    GLOBAL_HIDDEN_DIM = 256
    GLOBAL_LATENT_DIM = 64

    # Fusion Head
    FUSION_HIDDEN_DIM = 128
    DROPOUT_RATE = 0.1

    # Targets
    NUM_TARGETS = 2  # formation_energy_ev_natom, bandgap_energy_ev

    # ==========================================
    # Training Settings
    # ==========================================
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 200
    EARLY_STOPPING_PATIENCE = 20

    # Reproducibility
    SEED = 42

    # Compute
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """Creates necessary directories for the pipeline."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        print(f"Configured working directory: {cls.WORKING_DIR}")
        print(f"Configured submission directory: {cls.SUBMISSION_DIR}")
        print(f"Using device: {cls.DEVICE}")
