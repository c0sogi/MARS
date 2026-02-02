import os
import torch


class Config:
    """
    Configuration class for the Scalar Coupling Prediction task.
    This includes file paths, model hyperparameters, training settings,
    and chemical constants.
    """

    # ==========================================
    # 1. File Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_2"
    SUBMISSION_DIR = "./submission"

    # Ensure working and submission directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Raw Data Files
    STRUCTURES_CSV = os.path.join(INPUT_DIR, "structures.csv")
    DIPOLE_MOMENTS_CSV = os.path.join(INPUT_DIR, "dipole_moments.csv")
    POTENTIAL_ENERGY_CSV = os.path.join(INPUT_DIR, "potential_energy.csv")
    MULLIKEN_CHARGES_CSV = os.path.join(INPUT_DIR, "mulliken_charges.csv")

    # Metadata Files (Pre-split)
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output Files
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Cache Files
    PROCESSED_DATA_CACHE = os.path.join(WORKING_DIR, "processed_graphs.pt")
    STATS_CACHE = os.path.join(WORKING_DIR, "target_stats.npy")

    # ==========================================
    # 2. Chemical Constants & Mappings
    # ==========================================
    # Mapping atomic symbols to integers
    # QM9 dataset typically contains H, C, N, O, F
    ATOM_MAP = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}
    NUM_ATOM_TYPES = len(ATOM_MAP)

    # Mapping coupling types to integers
    COUPLING_TYPES = ["1JHC", "2JHH", "1JHN", "2JHN", "2JHC", "3JHH", "3JHC", "3JHN"]
    COUPLING_MAP = {ctype: i for i, ctype in enumerate(COUPLING_TYPES)}
    NUM_COUPLING_TYPES = len(COUPLING_TYPES)

    # ==========================================
    # 3. Data Processing Hyperparameters
    # ==========================================
    # Radius for graph construction (Angstroms)
    # 5.0A is generally sufficient to capture relevant chemical environments
    CUTOFF = 5.0

    # Maximum number of neighbors to consider per atom to manage memory
    MAX_NEIGHBORS = 32

    # Random Seed for reproducibility
    SEED = 42

    # ==========================================
    # 4. Model Hyperparameters (Directional MPNN)
    # ==========================================
    # Dimensionality of node/edge embeddings
    HIDDEN_CHANNELS = 128

    # Number of interaction blocks (layers)
    NUM_LAYERS = 4

    # Basis functions for geometry
    NUM_RBF = 16  # Radial Basis Functions count
    NUM_SBF = 16  # Spherical Basis Functions count
    ENVELOPE_EXPONENT = 5  # Polynomial exponent for envelope function

    # Output network
    NUM_OUTPUT_LAYERS = 3

    # Embedding dimensions
    ATOM_EMBEDDING_DIM = 64
    TYPE_EMBEDDING_DIM = 32

    # ==========================================
    # 5. Training Hyperparameters
    # ==========================================
    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # Optimization
    BATCH_SIZE = 64  # Adjust based on GPU memory (A100 40GB allows larger batches)
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5

    # Scheduler
    WARMUP_EPOCHS = 2
    LR_DECAY_FACTOR = 0.5
    LR_PATIENCE = 2
    MIN_LR = 1e-6

    # Training Loop
    MAX_EPOCHS = 20
    EARLY_STOPPING_PATIENCE = 5

    # Debugging / Development
    # Set to a small integer (e.g., 1000) to run on a subset of data for quick testing
    # Set to None to run on full dataset
    DEBUG_SAMPLE_SIZE = None

    def __init__(self):
        pass

    @staticmethod
    def print_config():
        print("=" * 40)
        print("CONFIG SETTINGS")
        print("=" * 40)
        print(f"Device: {Config.DEVICE}")
        print(f"Batch Size: {Config.BATCH_SIZE}")
        print(f"Learning Rate: {Config.LEARNING_RATE}")
        print(f"Max Epochs: {Config.MAX_EPOCHS}")
        print(f"Cutoff Radius: {Config.CUTOFF}")
        print(f"Hidden Channels: {Config.HIDDEN_CHANNELS}")
        print(f"Debug Sample Size: {Config.DEBUG_SAMPLE_SIZE}")
        print("=" * 40)
