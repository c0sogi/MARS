import os


class Config:
    """
    Configuration class for the Anisotropic Multi-Scale Physics-Aware Deep Sets (AMSP-DS) strategy.
    """

    # ==========================================
    # Directories and File Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_43"
    SUBMISSION_DIR = "./submission"

    # Ensure necessary directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata Files
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Input/Output Files
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache Files
    TRAIN_DATA_CACHE = os.path.join(WORKING_DIR, "train_data.npz")
    VAL_DATA_CACHE = os.path.join(WORKING_DIR, "val_data.npz")
    TEST_DATA_CACHE = os.path.join(WORKING_DIR, "test_data.npz")
    SCALERS_CACHE = os.path.join(WORKING_DIR, "scalers.npz")
    MODEL_CHECKPOINT = os.path.join(WORKING_DIR, "best_model.pt")

    # ==========================================
    # Feature Extraction Parameters
    # ==========================================
    # Neighbor counts for multi-scale context
    K_NEIGHBORS_SHORT = 6
    K_NEIGHBORS_MEDIUM = 24
    K_NEIGHBORS_PACKING = 12  # Used for packing ratio calculation

    # Physical Properties Dictionary
    # Format: "Symbol": [Atomic Mass (u), Covalent Radius (Angstrom), Electronegativity (Pauling)]
    ATOMIC_PROPS = {
        "Al": [26.981539, 1.21, 1.61],
        "Ga": [69.723, 1.22, 1.81],
        "In": [114.818, 1.42, 1.78],
        "O": [15.999, 0.66, 3.44],
    }

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    # Atomic Stream
    ATOMIC_HIDDEN_DIM = 512
    ATOMIC_LAYERS = 3

    # Global Stream
    GLOBAL_HIDDEN_DIM = 256
    GLOBAL_LAYERS = 2

    # Fusion Head
    FUSION_HIDDEN_DIM = 256

    # Regularization
    DROPOUT_RATE = 0.1

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 200
    PATIENCE = 20  # For Early Stopping

    # Reproducibility
    SEED = 42
