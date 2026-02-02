import os


class Config:
    """
    Configuration for the RBF-Augmented Dual-Stream Deep Sets model pipeline.
    """

    # Reproducibility
    SEED = 42

    # -------------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_3"
    SUBMISSION_DIR = "./submission"

    # Metadata files (generated previously)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Cache files for processed data (using .npz for efficiency)
    # Updated filenames to avoid loading incompatible cached data from previous runs
    TRAIN_CACHE_PATH = os.path.join(WORKING_DIR, "train_data_v2.npz")
    VAL_CACHE_PATH = os.path.join(WORKING_DIR, "val_data_v2.npz")
    TEST_CACHE_PATH = os.path.join(WORKING_DIR, "test_data_v2.npz")

    # Output paths
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pt")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # -------------------------------------------------------------------------
    # Data Preprocessing Parameters
    # -------------------------------------------------------------------------
    # Atomic species present in the dataset
    ATOM_TYPES = ["Al", "Ga", "In", "O"]
    NUM_ATOM_TYPES = len(ATOM_TYPES)

    # -------------------------------------------------------------------------
    # Model Architecture Hyperparameters
    # -------------------------------------------------------------------------
    # Atomic Stream (Enhanced Deep Sets)
    # Input vector: One-hot Atom Type (4) + Centered XYZ (3)
    # Removed RBF features to prevent overfitting (Cite solution_lesson_node_00012)
    ATOMIC_INPUT_DIM = NUM_ATOM_TYPES + 3
    ATOMIC_HIDDEN_DIM = 256
    ATOMIC_LATENT_DIM = 128  # Dimension of atomic features before pooling

    # Lattice Stream (Global Context)
    # Input vector: Lattice lengths (3) + Lattice angles (3) + Total atoms (1)
    LATTICE_INPUT_DIM = 7
    # Increased capacity to better capture global geometric constraints (Cite solution_lesson_node_00009)
    LATTICE_HIDDEN_DIMS = [256, 256]
    LATTICE_OUTPUT_DIM = 128

    # Fusion and Regressor
    # Aggregation: Concatenation of Global Mean Pool + Global Max Pool
    # Fusion Input: (Atomic_Latent * 2) + Lattice_Output
    FUSION_INPUT_DIM = (ATOMIC_LATENT_DIM * 2) + LATTICE_OUTPUT_DIM
    REGRESSOR_HIDDEN_DIMS = [256, 128, 64]
    OUTPUT_DIM = 2  # Targets: formation_energy_ev_natom, bandgap_energy_ev

    DROPOUT_RATE = 0.1

    # -------------------------------------------------------------------------
    # Training Parameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 150
    PATIENCE = 20  # For Early Stopping
    WEIGHT_DECAY = 1e-4  # Regularization

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------
    @classmethod
    def setup_directories(cls):
        """Creates necessary working directories if they don't exist."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize directories on module import
Config.setup_directories()
