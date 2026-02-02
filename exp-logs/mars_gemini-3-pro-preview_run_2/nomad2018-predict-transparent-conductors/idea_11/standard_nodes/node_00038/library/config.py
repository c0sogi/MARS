import os


class Config:
    """
    Configuration for the Multi-Scale Readout Crystal Graph Convolutional Network (MSR-CGCNN) experiment.
    Centralizes all file paths, model hyperparameters, and training settings.
    """

    # -------------------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------------------
    SEED = 42

    # -------------------------------------------------------------------------
    # File Paths
    # -------------------------------------------------------------------------
    # Input Data
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Working Directories (Idea Specific)
    WORKING_DIR = "./working/idea_11"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")

    # Output Paths
    # Note: Competition usually expects submission at ./submission/submission.csv
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Model Checkpoint
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")

    # Scaler Paths (saved for inference)
    TARGET_SCALER_PATH = os.path.join(CACHE_DIR, "target_scaler.pth")
    GLOBAL_SCALER_PATH = os.path.join(CACHE_DIR, "global_scaler.pth")

    # -------------------------------------------------------------------------
    # Data Hyperparameters
    # -------------------------------------------------------------------------
    # Neighbor graph construction
    NEIGHBOR_CUTOFF = 5.0  # Angstroms (Strict local constraint)
    MAX_NEIGHBORS = 12  # Limit edges per node for efficiency

    # Features
    ATOM_EMBEDDING_DIM = 64  # Embedding dimension for atomic numbers (nodes)

    # Global features to be extracted from metadata and processed via MLP
    GLOBAL_FEATURES = [
        "lattice_vector_1_ang",
        "lattice_vector_2_ang",
        "lattice_vector_3_ang",
        "lattice_angle_alpha_degree",
        "lattice_angle_beta_degree",
        "lattice_angle_gamma_degree",
        "number_of_total_atoms",
        "percent_atom_al",
        "percent_atom_ga",
        "percent_atom_in",
    ]

    # Targets
    TARGET_COLS = ["formation_energy_ev_natom", "bandgap_energy_ev"]

    # -------------------------------------------------------------------------
    # Model Architecture Hyperparameters
    # -------------------------------------------------------------------------
    # Backbone
    HIDDEN_CHANNELS = 128
    NUM_INTERACTION_LAYERS = 6  # Number of CGCNN layers

    # Edge Featurization
    NUM_RBF = 60  # Number of Gaussian Radial Basis Functions

    # Regularization
    DROPOUT_RATE = 0.1

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 48
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    MAX_EPOCHS = 150
    PATIENCE = 15  # Early stopping patience

    # -------------------------------------------------------------------------
    # Debugging / Development
    # -------------------------------------------------------------------------
    DEBUG = False  # Set to True to run on a small subset
    DEBUG_SAMPLE_SIZE = 100

    @classmethod
    def setup(cls):
        """Creates necessary working directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize directories immediately upon import
Config.setup()
