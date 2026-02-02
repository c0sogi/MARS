import os
import torch


class Config:
    """
    Configuration class for the Constraint-Aware Standardized Dual-Stream Network (CAS-DS Net).

    This class acts as a centralized repository for:
    1. File paths and directory structures.
    2. Data processing constants (normalization, image sizing).
    3. Model hyperparameters (architecture, dimensions).
    4. Training settings (learning rates, batch sizes, epochs).
    5. Metric-aligned constraints for loss calculation.
    """

    # ==========================================
    # 1. Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "train")
    TEST_IMG_DIR = os.path.join(INPUT_DIR, "test")

    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Working Directory for Outputs
    # We use 'idea_48' as the designated workspace for this iteration
    WORKING_DIR = "./working/idea_48"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")

    # Ensure output directories exist immediately upon config load
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # 2. Data Processing Hyperparameters
    # ==========================================
    IMG_SIZE = 260
    NUM_SLICES = 3  # Anchor slice + 2 boundary slices (2.5D representation)

    # Target Normalization Constants (Derived from Global Training Statistics)
    # These are used to standardize the target variable (FVC)
    # Mean: 2654.6528, Std: 801.7017 (from EDA)
    TARGET_MEAN = 2654.6528
    TARGET_STD = 801.7017

    # CT Scan Windowing (Lung Window)
    WINDOW_LEVEL = -600
    WINDOW_WIDTH = 1500

    # ==========================================
    # 3. Model Hyperparameters
    # ==========================================
    BACKBONE_NAME = "tf_efficientnet_b2_ns"  # EfficientNet-B2 with NoisyStudent weights
    IN_CHANNELS = 3  # 3 stacked slices treated as RGB channels
    FEATURE_DIM = 64  # Projection dimension for image features before fusion
    HIDDEN_DIM = 128  # Hidden dimension for MLP streams
    DROP_RATE = 0.0  # Explicitly set to 0.0 for Stream B (Residual Stream)

    # ==========================================
    # 4. Training Hyperparameters
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    BATCH_SIZE = 32
    EPOCHS = 50
    NUM_WORKERS = 4

    # Optimizer Settings
    LR_BACKBONE = 1e-4  # Lower LR for pre-trained backbone
    LR_HEAD = 1e-3  # Higher LR for randomly initialized heads
    WEIGHT_DECAY = 1e-2

    # Scheduler (Cosine Annealing)
    T_MAX = EPOCHS  # Cycle length matches total epochs
    ETA_MIN = 1e-6

    # Early Stopping
    PATIENCE = 10

    # ==========================================
    # 5. Metric & Loss Constraints
    # ==========================================
    # The competition metric clips sigma at 70ml and error at 1000ml
    MIN_SIGMA = 70.0
    MAX_ERROR = 1000.0

    # Standardized Minimum Sigma (Architectural Constraint)
    # We enforce the 70ml floor within the standardized optimization space.
    # Calculation: epsilon_std = 70 / sigma_global
    STD_MIN_SIGMA = MIN_SIGMA / TARGET_STD

    # ==========================================
    # 6. Debugging / Development
    # ==========================================
    DEBUG = False  # Set to True to run on a small subset for rapid testing
    DEBUG_SIZE = 100  # Number of samples to use when DEBUG is True
