import os
import torch


class Config:
    # -------------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_8"
    SUBMISSION_DIR = "./submission"

    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Cache files for processed data
    TRAIN_DATA_CACHE = os.path.join(WORKING_DIR, "train_data.npz")
    VAL_DATA_CACHE = os.path.join(WORKING_DIR, "val_data.npz")
    TEST_DATA_CACHE = os.path.join(WORKING_DIR, "test_data.npz")

    # Model checkpoint path
    MODEL_CHECKPOINT_PATH = os.path.join(WORKING_DIR, "best_model.pt")

    # Submission file
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Data / Feature Parameters
    # -------------------------------------------------------------------------
    # Atomic features: One-hot (4 elements) + Centered Coords (3) + NN Dist (1)
    ATOMIC_INPUT_DIM = 8

    # Global features: Lattice lengths (3) + Angles (3) + Volume (1) + Density (1) + Stoichiometry (3)
    GLOBAL_INPUT_DIM = 11

    # Symmetry features: Spacegroup ID (1)
    # Note: Spacegroups range from 1 to 230. We'll use an embedding layer.
    NUM_SPACEGROUPS = 231  # 0 (padding) + 1..230

    # Target variables
    TARGET_COLS = ["formation_energy_ev_natom", "bandgap_energy_ev"]
    NUM_TARGETS = 2

    # -------------------------------------------------------------------------
    # Model Hyperparameters
    # -------------------------------------------------------------------------
    # Atomic Stream (Residual Point Processor)
    ATOMIC_HIDDEN_DIM = 256
    NUM_RESIDUAL_BLOCKS = 4
    ATOMIC_DROPOUT = 0.1

    # Global Stream (Thermodynamic Context)
    GLOBAL_HIDDEN_DIM = 128
    GLOBAL_DROPOUT = 0.1

    # Symmetry Stream (Crystallographic Prior)
    SYMMETRY_EMBEDDING_DIM = 32

    # Fusion Head
    # Input to fusion is: (Atomic_Mean + Atomic_Max + Atomic_Std) * Hidden + Global + Symmetry
    # 3 * 256 + 128 + 32 = 768 + 128 + 32 = 928
    FUSION_HIDDEN_DIMS = [512, 256, 128]
    FUSION_DROPOUT = 0.1

    # -------------------------------------------------------------------------
    # Training Settings
    # -------------------------------------------------------------------------
    SEED = 42
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 200
    PATIENCE = 20  # Early stopping patience
    WEIGHT_DECAY = 1e-4

    # Scheduler
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 10
    SCHEDULER_MIN_LR = 1e-6

    # -------------------------------------------------------------------------
    # Device
    # -------------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def setup():
        """Ensures necessary directories exist."""
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)


# Initialize directories on import
Config.setup()
