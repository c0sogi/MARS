import os
import torch


class Config:
    """
    Configuration module for the RALI-Net (Residual-Augmented Latent Interaction Network) solution.
    Centralizes all file paths, hyperparameters, and constants.
    """

    # ==========================================
    # General & Reproducibility
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to use a subset of data for debugging

    # ==========================================
    # Paths
    # ==========================================
    # Input Data (ReadOnly)
    INPUT_DIR = "./input"
    TRAIN_DICOM_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DICOM_DIR = os.path.join(INPUT_DIR, "test")

    # Metadata (Pre-generated in previous steps)
    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Sample Submission
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Working Directories (Writeable)
    # Specific directory for this solution idea
    WORKING_DIR = "./working/idea_16"

    # Caching directory for processed data (numpy/parquet)
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # Checkpoints for model weights
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")

    # Final Submission directory
    SUBMISSION_DIR = "./submission"

    # File Paths for Outputs
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Preprocessing & Feature Engineering
    # ==========================================
    # Content-Adaptive Slice Selection
    NUM_SLICES = 3  # Anchor slice + 2 boundary slices

    # Image Resolution
    IMG_SIZE = 224  # Standard for EfficientNet-B0

    # Feature Scaling
    TIME_SCALE = 0.01  # Scale relative weeks (t_rel) by 0.01

    # Numerical Stability
    EPSILON = 1e-6

    # ==========================================
    # Model Architecture: RALI-Net
    # ==========================================
    BACKBONE_NAME = "efficientnet_b2"

    # Stream A: Over-Parameterized Clinical Residual
    # Input: FVC(1) + Time(1) + Age(1) + Sex(1) + Smoking(3 OHE) = 7
    CLINICAL_INPUT_DIM = 7
    CLINICAL_HIDDEN_DIM = 128

    # Stream B: Visual Interaction Stream
    # Visual Projection (from CNN backbone)
    VISUAL_PROJECTION_DIM = 128
    # Interaction Input: Visual(128) + Clinical(7)
    INTERACTION_INPUT_DIM = VISUAL_PROJECTION_DIM + CLINICAL_INPUT_DIM
    INTERACTION_HIDDEN_DIM = 256

    # Latent Fusion (Summation Space)
    LATENT_DIM = 128

    # Dropout Rate
    DROPOUT = 0.2

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    EPOCHS = 50
    BATCH_SIZE = 32  # A100 GPU allows larger batch size
    NUM_WORKERS = 4  # Data loading workers

    # Differential Learning Rates
    LR_BACKBONE = 1e-4  # Lower LR for fine-tuning CNN
    LR_HEAD = 1e-3  # Higher LR for MLPs

    # Regularization
    WEIGHT_DECAY = 1e-2

    # Scheduler (Cosine Annealing)
    T_MAX = EPOCHS
    ETA_MIN = 1e-6

    # Loss & Metric Constraints
    METRIC_CLIP_SIGMA = 70
    METRIC_MAX_ERROR = 1000

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """
        Creates the necessary working directories if they do not exist.
        """
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize directories upon module import
Config.setup()
