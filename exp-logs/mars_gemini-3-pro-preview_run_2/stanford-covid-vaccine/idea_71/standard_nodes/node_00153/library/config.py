import os
import torch


class Config:
    # =========================================================================
    # System & Paths
    # =========================================================================
    PROJECT_NAME = "RHI_GFN_Idea_71"
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Input Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working Directory (Idea Specific)
    WORKING_DIR = "./working/idea_71"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # File Paths
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Cache Files (Version Control for Preprocessing)
    # Using specific version tags to prevent stale cache usage
    TRAIN_CACHE = os.path.join(WORKING_DIR, "train_data_rhi_gfn_v1.npz")
    VAL_CACHE = os.path.join(WORKING_DIR, "val_data_rhi_gfn_v1.npz")
    TEST_CACHE = os.path.join(WORKING_DIR, "test_data_rhi_gfn_v1.npz")

    # Model Checkpoints & Outputs
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # =========================================================================
    # Data Specifications
    # =========================================================================
    SEQ_LENGTH = 107
    SEQ_SCORED = 68

    # Target Columns in the dataset
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # Columns used for scoring in the competition metric
    # Indices correspond to: reactivity (0), deg_Mg_pH10 (1), deg_Mg_50C (3)
    SCORED_COLS_INDICES = [0, 1, 3]
    NUM_TARGETS = 5

    # =========================================================================
    # Model Architecture (RHI-GFN)
    # =========================================================================
    # Input Features:
    # Sequence(4) + Structure(3) + LoopType(7) + PartnerIdentity(4) = 18
    IN_CHANNELS = 18

    # Hybrid Input Stem
    HYBRID_KERNEL_SIZE = 3

    # Main Backbone (Dense Dilated TCN)
    BACKBONE_GROWTH_RATE = 64
    BACKBONE_KERNEL_SIZE = 3
    # Dilations: Exponential growth for receptive field
    DILATIONS = [1, 2, 4, 8, 16, 32]
    DROPOUT = 0.1

    # Latent Representation
    LATENT_DIM = 64

    # Feedback Module
    FEEDBACK_IN_CHANNELS = 5  # Recycled predictions
    FEEDBACK_HIDDEN_CHANNELS = 32
    FEEDBACK_BACKBONE_GROWTH_RATE = 16
    FEEDBACK_LAYERS = 3  # Lightweight backbone for feedback

    # Interaction & Aggregation
    RNN_HIDDEN_DIM = 64  # Compact size to match feature dim
    RNN_LAYERS = 1
    RNN_BIDIRECTIONAL = True

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    # Small Batch Regime for better convergence on small data
    BATCH_SIZE = 16

    # Optimization
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4  # Standard regularization
    GRAD_CLIP = 1.0

    # Scheduler (ReduceLROnPlateau)
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 5

    # Iterative Refinement
    RECYCLING_STEPS = 1  # Number of feedback passes (Total passes = 1 + steps)
    AUX_LOSS_WEIGHT = 0.5  # Weight for the first pass loss

    # Training Loop
    EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 10
    NUM_WORKERS = 2  # Adjusted for available vCPUs

    # Debugging
    DEBUG = False
    DEBUG_SUBSET_SIZE = 100  # Number of samples to use if DEBUG is True

    @classmethod
    def print_config(cls):
        """Prints the current configuration."""
        print(f"--- {cls.PROJECT_NAME} Configuration ---")
        print(f"Device: {cls.DEVICE}")
        print(f"Working Dir: {cls.WORKING_DIR}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        print(f"Backbone Growth Rate: {cls.BACKBONE_GROWTH_RATE}")
        print(f"Recycling Steps: {cls.RECYCLING_STEPS}")
        print("-" * 40)
