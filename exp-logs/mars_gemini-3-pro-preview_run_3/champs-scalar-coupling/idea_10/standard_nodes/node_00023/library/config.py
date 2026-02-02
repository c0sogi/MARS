import os
import torch


class Config:
    # ==========================================
    # File Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_10"
    SUBMISSION_DIR = "./submission"

    # Ensure working and submission directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Raw Data Files
    STRUCTURES_CSV = os.path.join(INPUT_DIR, "structures.csv")
    DIPOLE_MOMENTS_CSV = os.path.join(INPUT_DIR, "dipole_moments.csv")
    MAGNETIC_SHIELDING_CSV = os.path.join(INPUT_DIR, "magnetic_shielding_tensors.csv")
    MULLIKEN_CHARGES_CSV = os.path.join(INPUT_DIR, "mulliken_charges.csv")
    POTENTIAL_ENERGY_CSV = os.path.join(INPUT_DIR, "potential_energy.csv")
    SCALAR_COUPLING_CONTRIBUTIONS_CSV = os.path.join(
        INPUT_DIR, "scalar_coupling_contributions.csv"
    )

    # Metadata Files (Pre-split)
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Processed Data Cache Paths
    PROCESSED_DATA_DIR = os.path.join(WORKING_DIR, "processed")
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    # Model Checkpoint
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Global Settings
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # For data loading

    # ==========================================
    # Data Processing Hyperparameters
    # ==========================================
    # Atom mapping (H, C, N, O, F are the only elements in this dataset)
    ATOM_MAP = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}
    NUM_ATOM_TYPES = len(ATOM_MAP)

    # Coupling Type Mapping (Sorted alphabetically)
    COUPLING_TYPES = ["1JHC", "1JHN", "2JHC", "2JHH", "2JHN", "3JHC", "3JHH", "3JHN"]
    TYPE_MAP = {t: i for i, t in enumerate(COUPLING_TYPES)}
    NUM_COUPLING_TYPES = len(COUPLING_TYPES)

    # Graph Construction
    CUTOFF_RADIUS = 5.0  # Angstroms
    MAX_NEIGHBORS = 32  # Max neighbors per node to consider

    # ==========================================
    # Model Architecture Hyperparameters
    # ==========================================
    # Backbone: Directional Message Passing
    HIDDEN_DIM = 128
    NUM_LAYERS = 4  # Number of message passing interaction blocks
    NUM_RBF = 64  # Number of RBF kernels for distance expansion
    NUM_ANGLE_RBF = 32  # Number of RBF kernels for angular expansion (cos theta)

    # Readout
    TYPE_EMB_DIM = 64  # Dimension for coupling type embedding
    OUTPUT_HEADS = 1  # Single shared head

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 128  # Number of molecules per batch
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 1e-5
    MAX_EPOCHS = 30
    PATIENCE = 5  # Early stopping patience

    # Scheduler
    SCHEDULER_T_0 = 10  # Cosine Annealing restart interval
    SCHEDULER_T_MULT = 2

    # Loss Weights
    # Total Loss = L_coupling + AUX_WEIGHT * (L_shielding + L_charge)
    AUX_LOSS_WEIGHT = 0.1

    # Debugging
    DEBUG = False  # Set to True to run on a small subset
    DEBUG_SAMPLE_SIZE = 2000

    # ==========================================
    # Normalization Stats (Placeholders)
    # ==========================================
    # These will be populated during the data processing phase or hardcoded if known
    # Format: {type_index: (mean, std)}
    COUPLING_STATS = {}

    # Auxiliary target stats (mean, std)
    SHIELDING_MEAN = 0.0
    SHIELDING_STD = 1.0
    CHARGE_MEAN = 0.0
    CHARGE_STD = 1.0

    def __str__(self):
        return str(self.__class__.__dict__)
