import os
import torch


class Config:
    """
    Configuration for the Normalized Shared-Latent Holistic Network (NSL-HN).
    Centralizes all hyperparameters, paths, and constants.
    """

    # ==========================================
    # 1. Experiment & System Settings
    # ==========================================
    PROJECT_NAME = "lung_decline_prediction"
    EXP_ID = "idea_53"
    SEED = 42
    DEBUG = False  # Set True to run on a small subset for debugging

    # Compute
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Adjust based on vCPU count (12 available)

    # ==========================================
    # 2. File Paths
    # ==========================================
    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Input Data
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    DICOM_TRAIN_DIR = os.path.join(INPUT_ROOT, "train")
    DICOM_TEST_DIR = os.path.join(INPUT_ROOT, "test")

    # Output & Caching
    # Cache directory for preprocessed tri-slab arrays
    CACHE_DIR = os.path.join(WORKING_DIR, EXP_ID)
    # Path to save the best model checkpoint
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, f"{EXP_ID}_best_model.pth")
    # Submission output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # 3. Data Processing (Tri-Slab)
    # ==========================================
    IMG_SIZE = 224  # Native resolution for EfficientNet-B0
    NUM_SLICES = 3  # Tri-slab configuration
    OVERLAP_RATIO = 0.15  # 15% overlap between slabs

    # Views to generate
    USE_AXIAL = True
    USE_CORONAL = True

    # Normalization (Hounsfield Units)
    HU_MIN = -1000
    HU_MAX = 400

    # ==========================================
    # 4. Model Architecture (NSL-HN)
    # ==========================================
    BACKBONE_NAME = "efficientnet_b0"
    BACKBONE_DIM = 1280  # Native output dim of B0 (no projection)

    # Shared Latent Topology
    LATENT_DIM = 128  # Dimension for T_lat (Shared Latent Vector)
    HIDDEN_DIM = 256  # Hidden dim for the tabular MLP

    # Tabular Features for Shared Encoder
    # Note: 'Weeks' is strictly excluded from the static encoder input
    TABULAR_FEATURES = ["Percent", "Age", "Sex", "SmokingStatus"]

    # Head Configuration
    DROPOUT_RATE = 0.2

    # ==========================================
    # 5. Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 32
    EPOCHS = 50
    LR = 1e-4
    WEIGHT_DECAY = 1e-2

    # Scheduler (Cosine Annealing)
    T_MAX = 50
    ETA_MIN = 1e-6

    # Early Stopping
    PATIENCE = 8

    # ==========================================
    # 6. Metric & Inference
    # ==========================================
    # Modified Laplace Log Likelihood constants
    CONFIDENCE_CLIP = 70.0  # sigma_clipped min value
    MAX_ERROR = 1000.0  # Delta max value

    @classmethod
    def setup(cls):
        """Creates necessary output directories."""
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

    @classmethod
    def get_transforms(cls):
        """
        Returns augmentation configuration.
        Note: Intensity augmentations are strictly disabled.
        """
        # This is a placeholder for configuration values used by dataset class
        return {
            "horizontal_flip_prob": 0.5,
            "shift_limit": 0.05,
            "rotate_limit": 10,
            "brightness_contrast": False,  # Strictly disabled
        }
