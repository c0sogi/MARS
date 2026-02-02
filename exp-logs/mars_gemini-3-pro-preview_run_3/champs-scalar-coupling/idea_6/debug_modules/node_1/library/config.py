import os
import torch


class Config:
    """
    Configuration class for the Scalar Coupling Prediction Task.
    Implements the 'Physics-Calibrated Dual-Graph Network' strategy settings.
    """

    # ==========================================
    # Global Settings
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to use a small subset of data for debugging
    DEBUG_SAMPLE_SIZE = 2000  # Number of molecules to use when DEBUG is True

    # ==========================================
    # Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_6"
    SUBMISSION_DIR = "./submission"

    # Ensure necessary writeable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata Files (Pre-generated)
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Raw Data Files
    STRUCTURES_CSV = os.path.join(INPUT_DIR, "structures.csv")
    STRUCTURES_DIR = os.path.join(INPUT_DIR, "structures")
    DIPOLE_MOMENTS_CSV = os.path.join(INPUT_DIR, "dipole_moments.csv")
    MAGNETIC_SHIELDING_CSV = os.path.join(INPUT_DIR, "magnetic_shielding_tensors.csv")
    MULLIKEN_CHARGES_CSV = os.path.join(INPUT_DIR, "mulliken_charges.csv")
    POTENTIAL_ENERGY_CSV = os.path.join(INPUT_DIR, "potential_energy.csv")
    SCALAR_CONTRIBUTIONS_CSV = os.path.join(
        INPUT_DIR, "scalar_coupling_contributions.csv"
    )
    SAMPLE_SUBMISSION_CSV = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Processed/Cached Data Paths
    # We use a subdirectory for caching processed graph data to keep working dir clean
    PROCESSED_CACHE_DIR = os.path.join(WORKING_DIR, "processed")
    os.makedirs(PROCESSED_CACHE_DIR, exist_ok=True)

    # Path to store calculated statistics (mean/std) for targets
    STATS_PATH = os.path.join(WORKING_DIR, "target_stats.npy")

    # Output Paths
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Constants
    # ==========================================
    # Atom types found in the dataset
    ATOM_TYPES = ["H", "C", "N", "O", "F"]

    # Coupling types to predict
    COUPLING_TYPES = ["1JHC", "2JHC", "3JHC", "1JHN", "2JHN", "3JHN", "2JHH", "3JHH"]

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    # Dual-Graph Architecture Settings
    HIDDEN_DIM = 256
    NUM_LAYERS = 4  # Number of interaction blocks (Atom <-> Line Graph updates)
    NUM_HEADS = 8  # Number of output heads (one per coupling type)
    DROPOUT = 0.1

    # Geometric Feature Settings
    RBF_CUTOFF = 5.0  # Angstroms (Interaction radius for graph construction)
    NUM_RBF = 128  # Number of Gaussian RBFs for distance expansion
    NUM_ANGLE_RBF = 32  # Number of Gaussian RBFs for angular cosine expansion

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    # Compute Settings (A100-40GB allows for larger batches)
    BATCH_SIZE = 128

    # Optimization
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    MAX_EPOCHS = 30  # Sufficient for convergence given the dataset size

    # Physics Constraints
    # Lambda weighting for auxiliary losses (Shielding + Charges)
    # Set to 0.1 to prevent auxiliary tasks from dominating the gradient
    AUX_LOSS_WEIGHT = 0.1

    # Scheduler Settings (Cosine Annealing / OneCycle)
    PCT_START = 0.3
    DIV_FACTOR = 25
    FINAL_DIV_FACTOR = 1000

    # Hardware Configuration
    NUM_WORKERS = 12  # Utilizing available vCPUs
    PIN_MEMORY = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
