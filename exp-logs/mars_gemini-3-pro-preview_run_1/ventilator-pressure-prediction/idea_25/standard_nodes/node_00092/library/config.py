import os
import torch


class Config:
    """
    Global configuration for the Ventilator Pressure Prediction pipeline.
    Implements the 'Feature-Complete Uniform-Capacity Physics-Composite Network' strategy.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    EXPERIMENT_ID = "idea_25"

    # =========================================================================
    # File Paths & Directories
    # =========================================================================
    # Input Metadata (Pre-split)
    INPUT_DIR = "./metadata"
    TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
    VAL_CSV = os.path.join(INPUT_DIR, "val.csv")
    TEST_CSV = os.path.join(INPUT_DIR, "test.csv")
    SAMPLE_SUBMISSION = "./input/sample_submission.csv"

    # Working Directory (Outputs)
    WORKING_DIR = os.path.join("./working", EXPERIMENT_ID)
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    MODEL_PATH = os.path.join(WORKING_DIR, "model.pth")

    # Submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # Compute
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
    PIN_MEMORY = True

    # =========================================================================
    # Feature Engineering Configuration
    # =========================================================================
    # Feature Completeness: Include u_out in the input features
    INCLUDE_U_OUT_IN_INPUT = True

    # Physics Fidelity: Generate interaction terms (R*u_in, Vol/C)
    USE_PHYSICS_FEATURES = True

    # Temporal Dynamics
    USE_LAGS = True
    LAG_STEPS = [1, 2, 3, 4]
    USE_DIFFS = True  # First and second differences of u_in

    # Scaling
    ROBUST_SCALER_QUANTILE_RANGE = (5.0, 95.0)

    # =========================================================================
    # Model Architecture (Uniform-Capacity Physics-Composite)
    # =========================================================================
    # Uniform width backbone to maintain representational capacity
    HIDDEN_DIM = 512

    # Stem configuration
    STEM_KERNEL_SIZES = [3, 5, 7]

    # Backbone configuration
    NUM_LAYERS = 4
    FFN_EXPANSION_FACTOR = 2  # Expands to 1024 internally
    DROPOUT = 0.1

    # Residuals
    USE_STRICT_IDENTITY = True  # No weights on skip connections where dims match

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    EPOCHS = 35  # Extended horizon for convergence
    BATCH_SIZE = 512  # High batch size for stable updates

    # Optimizer (AdamW)
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2

    # Scheduler (OneCycleLR)
    PCT_START = 0.3
    DIV_FACTOR = 25.0
    FINAL_DIV_FACTOR = 10000.0

    # Gradient Clipping (Strict requirement for LSTM stability)
    CLIP_GRAD = 1.0

    # Loss Function
    AUX_WEIGHT = 0.3  # Weight for the auxiliary head on Block 2
    MASK_EXPIRATORY_PHASE = True  # Zero out loss where u_out=1

    @classmethod
    def print_config(cls):
        """Prints the current configuration."""
        print(f"=== Configuration: {cls.EXPERIMENT_ID} ===")
        print(f"Device: {cls.DEVICE}")
        print(f"Epochs: {cls.EPOCHS}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Hidden Dim: {cls.HIDDEN_DIM}")
        print(f"Clip Grad: {cls.CLIP_GRAD}")
        print(f"Include u_out: {cls.INCLUDE_U_OUT_IN_INPUT}")
        print(f"Working Dir: {cls.WORKING_DIR}")
        print("==========================================")
