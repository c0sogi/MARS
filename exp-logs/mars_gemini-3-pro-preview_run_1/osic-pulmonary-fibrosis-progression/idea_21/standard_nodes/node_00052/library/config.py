import os
import torch


class Config:
    """
    Central configuration for the Clinically-Gated Symmetric Dual-Axis Network pipeline.
    """

    # ==========================================
    # 1. Environment & Paths
    # ==========================================
    # Root directories
    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Input Data Paths (from Metadata)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_ROOT, "sample_submission.csv")

    # Output & Cache Paths
    # idea_21 is the designated folder for this iteration's artifacts
    IDEA_DIR = os.path.join(WORKING_DIR, "idea_21")
    CACHE_DIR = IDEA_DIR
    CHECKPOINT_DIR = os.path.join(IDEA_DIR, "checkpoints")
    MODEL_SAVE_PATH = os.path.join(IDEA_DIR, "best_model.pth")

    # Submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # 2. Global Settings
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Debugging flags
    DEBUG = False  # Set to True to run on a small subset
    DEBUG_DATA_SIZE = 50  # Number of patients to use in debug mode

    # ==========================================
    # 3. Data Preprocessing (Tri-Slab)
    # ==========================================
    IMG_SIZE = 224

    # Tri-Slab Generation Constants
    SLAB_COUNT = 3
    SLAB_OVERLAP = 0.15  # 15% overlap between slabs

    # Tabular Data Configuration
    # Features to be extracted and encoded
    NUMERICAL_FEATURES = ["Weeks", "Percent", "Age"]
    CATEGORICAL_FEATURES = ["Sex", "SmokingStatus"]

    # ==========================================
    # 4. Model Architecture
    # ==========================================
    BACKBONE_NAME = "efficientnet_b0"
    BACKBONE_PRETRAINED = True

    # Dimensionality
    VISUAL_FEATURE_DIM = 1280  # Native output dim of EfficientNet-B0 GAP
    TABULAR_HIDDEN_DIM = 1280  # Projected dimension for tabular data to match visual

    # ==========================================
    # 5. Training Hyperparameters
    # ==========================================
    N_EPOCHS = 50
    BATCH_SIZE = 32
    NUM_WORKERS = 4

    # Optimization
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2

    # Scheduler (Cosine Annealing)
    T_MAX = N_EPOCHS  # Cycle length for Cosine Annealing
    ETA_MIN = 1e-6  # Minimum learning rate

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 8

    # ==========================================
    # 6. Metric & Loss Constants
    # ==========================================
    # Modified Laplace Log Likelihood constraints
    ERROR_MAX_THRESHOLD = 1000.0  # Clip absolute error at 1000 ml
    CONFIDENCE_MIN_THRESHOLD = 70.0  # Clip confidence (sigma) at 70 ml

    @classmethod
    def setup(cls):
        """Ensures all necessary working directories exist."""
        os.makedirs(cls.IDEA_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize environment directories
Config.setup()
