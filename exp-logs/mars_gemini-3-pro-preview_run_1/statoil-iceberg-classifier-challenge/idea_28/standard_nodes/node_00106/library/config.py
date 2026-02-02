import os
import torch


class Config:
    """
    Configuration class for the Trajectory-Replayed ResNet-18 Ensemble.
    Serves as a shared state object for hyperparameters, paths, and dynamic training states.
    """

    # =========================================================================
    # Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_28"

    # Sub-directories for artifacts
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"

    # Source Files
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")

    # Metadata Files (Pre-generated)
    TRAIN_META = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_META = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Submission Output
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Processing & Augmentation
    # =========================================================================
    IMG_SIZE = 224
    NUM_CLASSES = 1

    # Global Min-Max Normalization Statistics (derived from analysis)
    # Band 1 (HH) Range: approx [-45, 35]
    # Band 2 (HV) Range: approx [-45, 20]
    # We use these fixed bounds to normalize inputs to [0, 1]
    BAND1_MIN = -45.0
    BAND1_MAX = 35.0
    BAND2_MIN = -45.0
    BAND2_MAX = 20.0

    # Augmentation Hyperparameters
    AUG_ROTATE_90_PROB = 0.5
    AUG_SHIFT_SCALE_ROTATE_PROB = 0.5
    AUG_SHIFT_LIMIT = 0.0625
    AUG_SCALE_LIMIT = 0.1
    AUG_ROTATE_LIMIT = 20
    AUG_HFLIP_PROB = 0.5
    AUG_VFLIP_PROB = 0.5

    # DataLoader
    NUM_WORKERS = 2
    PIN_MEMORY = True

    # =========================================================================
    # Model Architecture
    # =========================================================================
    MODEL_NAME = "resnet18"
    PRETRAINED = True
    DROPOUT_RATE = 0.5

    # =========================================================================
    # Training - General
    # =========================================================================
    SEED = 42
    BATCH_SIZE = 64
    LABEL_SMOOTHING = 0.05

    # Optimizer: AdamW
    OPTIMIZER_LR = 2e-4
    OPTIMIZER_WEIGHT_DECAY = 0.01

    # =========================================================================
    # Phase 1: Calibration (Trajectory Discovery)
    # =========================================================================
    # Used to find optimal epochs and scheduler milestones
    PHASE1_MAX_EPOCHS = 50
    PHASE1_PATIENCE = 10
    PHASE1_FACTOR = 0.5
    PHASE1_MIN_LR = 1e-6

    # =========================================================================
    # Phase 2: Production (Trajectory Replay + Cyclic SWA)
    # =========================================================================
    # These parameters are placeholders and will be updated via update_production_params
    # after Phase 1 determines the optimal trajectory.
    PHASE2_TOTAL_EPOCHS = 30  # Default Placeholder
    PHASE2_MILESTONES = []  # Default Placeholder
    PHASE2_FINAL_LR = 1e-5  # Default Placeholder

    # Low-Energy Cyclic SWA Settings
    # SWA starts AFTER the main trajectory replay is finished
    SWA_START_EPOCH = -1  # Will be set to PHASE2_TOTAL_EPOCHS
    SWA_CYCLES = 3
    SWA_CYCLE_LEN = 4
    SWA_LR_MAX = 1e-5  # Will be set to PHASE2_FINAL_LR
    SWA_LR_MIN = 5e-6  # Will be set to PHASE2_FINAL_LR / 2

    # =========================================================================
    # System & Debug
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DEBUG = False
    DEBUG_SAMPLES = 100  # Number of samples to use if DEBUG is True

    @classmethod
    def create_directories(cls):
        """Creates the necessary directory structure for artifacts."""
        dirs = [cls.WORKING_DIR, cls.CACHE_DIR, cls.CHECKPOINT_DIR, cls.SUBMISSION_DIR]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
        # print(f"Directories created at {cls.WORKING_DIR}")

    @classmethod
    def update_production_params(cls, optimal_epochs, milestones, final_lr):
        """
        Updates the Phase 2 configuration based on the trajectory discovered in Phase 1.

        Args:
            optimal_epochs (int): The epoch where Phase 1 converged (best val loss).
            milestones (list): List of epochs where the scheduler reduced the LR.
            final_lr (float): The learning rate at the point of convergence.
        """
        cls.PHASE2_TOTAL_EPOCHS = optimal_epochs
        cls.PHASE2_MILESTONES = milestones
        cls.PHASE2_FINAL_LR = final_lr

        # Configure SWA to start after the main trajectory replay
        cls.SWA_START_EPOCH = optimal_epochs
        cls.SWA_LR_MAX = final_lr
        cls.SWA_LR_MIN = final_lr / 2.0

        print(
            f"Config updated for Phase 2: Epochs={optimal_epochs}, Milestones={milestones}, Final LR={final_lr}"
        )

    @classmethod
    def to_dict(cls):
        """Returns a dictionary representation of the configuration."""
        return {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("__") and not callable(v)
        }
