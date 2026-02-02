import os
import torch


class Config:
    """
    Global configuration for the Metric-Aligned Cascaded Latent-Interaction Network (MACLI-Net).
    """

    # --------------------------------------------------------------------------
    # Experiment Setup
    # --------------------------------------------------------------------------
    SEED = 42
    DEBUG = False  # Set to True to use a subset of data for debugging
    EXPERIMENT_NAME = "idea_dsprnet"

    # --------------------------------------------------------------------------
    # Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = os.path.join("./working", EXPERIMENT_NAME)

    # Metadata Files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Working Directories
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")

    # Output Files
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Data & Preprocessing
    # --------------------------------------------------------------------------
    # Image parameters
    MODEL_NAME = "efficientnet_b2"
    IMG_SIZE = 260  # Native resolution for EfficientNet-B2
    NUM_SLICES = 3  # Anchor + 2 boundaries
    SLICE_THRESHOLD = 0.5  # Area threshold for boundary slices

    # Clinical features
    # Features: Baseline FVC, Baseline Percent, Age, Sex, Smoking, Relative Time
    CLINICAL_INPUT_DIM = 6

    # --------------------------------------------------------------------------
    # Model Architecture
    # --------------------------------------------------------------------------
    PRETRAINED = True
    IMG_EMBED_DIM = 64
    CLINICAL_HIDDEN_DIM = 128
    CLINICAL_LATENT_DIM = 64

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    EPOCHS = 50
    BATCH_SIZE = 32
    NUM_WORKERS = 4

    # Optimization
    LR_BACKBONE = 1e-4
    LR_HEAD = 1e-3
    WEIGHT_DECAY = 1e-2

    # Scheduler (Cosine Annealing)
    T_MAX = EPOCHS
    ETA_MIN = 1e-6

    # Early Stopping
    PATIENCE = 10

    # --------------------------------------------------------------------------
    # Metric Constants
    # --------------------------------------------------------------------------
    # Metric: - (sqrt(2) * delta / sigma_clipped) - ln(sqrt(2) * sigma_clipped)
    METRIC_CLIP_SIGMA = 70.0
    METRIC_MAX_ERROR = 1000.0

    # --------------------------------------------------------------------------
    # Compute
    # --------------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """
        Creates the necessary directory structure for the experiment.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Automatically setup directories when config is imported
Config.setup()
