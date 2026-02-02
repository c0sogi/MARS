import os
import torch


class Config:
    """
    Configuration for Global-Bonding Augmented Multi-Scale Deep Sets (GBA-MS-DS).
    Acts as a central repository for paths, hyperparameters, physical constants,
    and training settings.
    """

    # ==========================================
    # Directories and Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_49"
    SUBMISSION_DIR = "./submission"

    # Specific file paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure writable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Cache Files (using .npz for efficient storage)
    TRAIN_CACHE_FILE = os.path.join(WORKING_DIR, "train_data.npz")
    VAL_CACHE_FILE = os.path.join(WORKING_DIR, "val_data.npz")
    TEST_CACHE_FILE = os.path.join(WORKING_DIR, "test_data.npz")
    SCALER_CACHE_FILE = os.path.join(WORKING_DIR, "scalers.npz")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pt")

    # ==========================================
    # Physical Constants & Mappings
    # ==========================================
    # Elements present in the dataset
    ATOM_LIST = ["Al", "Ga", "In", "O"]
    ATOM_TO_IDX = {"Al": 0, "Ga": 1, "In": 2, "O": 3}
    NUM_ATOM_TYPES = len(ATOM_LIST)

    # Element Pairs for Global Bond Statistics (Upper triangular indices)
    # Pairs: (Al,Al), (Al,Ga), (Al,In), (Al,O), (Ga,Ga), (Ga,In), (Ga,O), (In,In), (In,O), (O,O)
    BOND_PAIRS = [
        ("Al", "Al"),
        ("Al", "Ga"),
        ("Al", "In"),
        ("Al", "O"),
        ("Ga", "Ga"),
        ("Ga", "In"),
        ("Ga", "O"),
        ("In", "In"),
        ("In", "O"),
        ("O", "O"),
    ]
    NUM_BOND_PAIRS = len(BOND_PAIRS)

    # Physical Properties (Mass in u, Radius in pm, Electronegativity Pauling)
    # Radius values are empirical atomic radii
    ATOMIC_MASS = {"Al": 26.9815, "Ga": 69.723, "In": 114.818, "O": 15.999}
    ATOMIC_RADIUS = {"Al": 143.0, "Ga": 135.0, "In": 167.0, "O": 60.0}
    ELECTRONEGATIVITY = {"Al": 1.61, "Ga": 1.81, "In": 1.78, "O": 3.44}

    # ==========================================
    # Feature Engineering Parameters
    # ==========================================
    # Multi-Scale Context Neighbors
    K_NEAR = 6
    K_FAR = 24

    # Atomic Feature Dimensions
    # 4 (One-hot ID) + 3 (Coords) + 1 (NN Dist) + 1 (Packing Ratio) + 4 (Ctx K_NEAR) + 4 (Ctx K_FAR)
    ATOMIC_FEATURE_DIM = 17

    # Global Feature Dimensions
    # 3 (Lattice Vecs) + 3 (Angles) + 1 (Vol) + 1 (Density) + 4 (Stoich) + 1 (N_atoms)
    # + 3 (Aspect Ratios) + 3 (Weighted Physics) + 10 (Bond Stats)
    GLOBAL_FEATURE_DIM = 29

    # ==========================================
    # Model Architecture
    # ==========================================
    # Atomic Stream (Wide MLP)
    ATOMIC_HIDDEN_DIM = 512
    ATOMIC_LAYERS = 3
    ATOMIC_DROPOUT = 0.1

    # Global Stream (High-Capacity MLP)
    GLOBAL_HIDDEN_DIM = 256
    GLOBAL_DROPOUT = 0.1

    # Fusion Head
    FUSION_HIDDEN_DIM = 256
    OUTPUT_DIM = 2  # formation_energy_ev_natom, bandgap_energy_ev

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Training loop
    EPOCHS = 200
    BATCH_SIZE = 32  # Number of graphs per batch (sparse batching)
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Early Stopping
    PATIENCE = 20
    MIN_DELTA = 1e-6

    # Scheduler (ReduceLROnPlateau)
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 5
    SCHEDULER_MIN_LR = 1e-6

    # ==========================================
    # Data Processing Flags
    # ==========================================
    # Columns to scale (StandardScaler)
    # Note: Atomic Identity (indices 0-3 of atomic features) are NOT scaled.
    # All global features are scaled.
    SCALE_ATOMIC_CONTINUOUS = True
    SCALE_GLOBAL = True
