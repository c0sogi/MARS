import os
import torch


class Config:
    """
    Configuration class for the Scalable Physics-Regularized Continuous Filter Network (SP-CFN).
    Centralizes all hyperparameters, paths, and constants.
    """

    # ==========================================
    # General Setup
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging

    # ==========================================
    # Directories & Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_14"
    SUBMISSION_DIR = "./submission"

    # Ensure working and submission directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Output Files
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Input Metadata
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Raw Data Files
    STRUCTURES_CSV = os.path.join(INPUT_DIR, "structures.csv")
    DIPOLE_MOMENTS_CSV = os.path.join(INPUT_DIR, "dipole_moments.csv")
    MAGNETIC_SHIELDING_CSV = os.path.join(INPUT_DIR, "magnetic_shielding_tensors.csv")
    MULLIKEN_CHARGES_CSV = os.path.join(INPUT_DIR, "mulliken_charges.csv")
    POTENTIAL_ENERGY_CSV = os.path.join(INPUT_DIR, "potential_energy.csv")
    SCALAR_COUPLING_CONTRIBUTIONS_CSV = os.path.join(
        INPUT_DIR, "scalar_coupling_contributions.csv"
    )

    # ==========================================
    # Data Processing & Physics
    # ==========================================
    # Cutoff radius for generating the molecular graph (Angstroms)
    CUTOFF = 5.0

    # Atom Type Mapping (H, C, N, O, F are the elements in this dataset)
    ATOM_MAP = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}
    NUM_ATOM_TYPES = len(ATOM_MAP)

    # Coupling Type Mapping
    COUPLING_TYPES = ["1JHC", "2JHC", "3JHC", "1JHN", "2JHN", "3JHN", "2JHH", "3JHH"]
    TYPE_MAP = {t: i for i, t in enumerate(COUPLING_TYPES)}
    NUM_COUPLING_TYPES = len(COUPLING_TYPES)

    # ==========================================
    # Model Architecture
    # ==========================================
    # Node embedding dimension
    HIDDEN_DIM = 128

    # Number of continuous filter interaction blocks
    NUM_INTERACTIONS = 4

    # Radial Basis Function (RBF) settings
    NUM_RBF = 50
    RBF_START = 0.0
    RBF_END = 5.0

    # Output head settings
    # Whether to use a shared head or separate heads is handled by architecture logic,
    # but we define dimensions here.

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 512  # A100 40GB can handle large batches
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 1e-5

    # Scheduler settings (Cosine Annealing Warm Restarts)
    T_0 = 10
    T_MULT = 2
    ETA_MIN = 1e-6

    # Training duration
    MAX_EPOCHS = 30
    PATIENCE = 5  # For Early Stopping

    # Loss Balancing
    # Weight for auxiliary losses (Shielding + Charge) relative to main coupling loss
    LAMBDA_AUX = 0.1

    # Hardware
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def get_structure_file_path(molecule_name):
        """Returns the path to the xyz file for a given molecule."""
        return os.path.join(Config.INPUT_DIR, "structures", f"{molecule_name}.xyz")
