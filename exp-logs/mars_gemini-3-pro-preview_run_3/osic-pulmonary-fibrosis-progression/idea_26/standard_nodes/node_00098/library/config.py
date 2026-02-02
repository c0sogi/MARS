import os
import torch


class Config:
    # Reproducibility
    SEED = 42

    # File Paths
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Working Directories (for caching and checkpoints)
    # Using 'idea_27' as the specific identifier for this experiment iteration
    WORKING_DIR = "./working/idea_27"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Data Preprocessing
    # Radiological Windowing (Lung Window)
    WINDOW_LEVEL = -600
    WINDOW_WIDTH = 1500

    # Slice Selection
    NUM_SLICES = 3  # Anchor + 2 boundaries
    IMG_SIZE = 260  # Native resolution for EfficientNet-B2

    # Feature Engineering
    TIME_SCALE = 0.01  # Scale relative weeks by 0.01

    # Model Architecture
    BACKBONE_NAME = "efficientnet_b2"
    IMG_EMBED_DIM = 64
    CLINICAL_HIDDEN_DIM = 128
    VISUAL_HIDDEN_DIM = 128
    LATENT_DIM = 64

    # Training Hyperparameters
    BATCH_SIZE = 32
    EPOCHS = 50

    # Differential Learning Rates
    LR_BACKBONE = 1e-4
    LR_HEAD = 1e-3

    # Scheduler
    T_MAX = 50  # For Cosine Annealing
    ETA_MIN = 1e-6

    # Loss / Metric
    # Metric-Aligned Laplace Log Likelihood constants
    # We don't hardcode the clip here for the loss, but for post-processing
    MIN_UNCERTAINTY = 70.0
    MAX_ERROR = 1000.0

    # Dataset Statistics (Moved here to avoid circular imports)
    STATS = {
        "fvc_mean": 2654.65,
        "fvc_std": 801.70,
        "age_mean": 67.58,
        "age_std": 6.62,
        "percent_mean": 76.91,
        "percent_std": 19.19,
    }

    @staticmethod
    def get_device():
        """Returns the appropriate device (GPU if available, else CPU)."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @classmethod
    def setup(cls):
        """Ensures necessary working directories exist."""
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Automatically setup directories when imported
Config.setup()
