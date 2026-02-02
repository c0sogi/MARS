import os
import torch


class Config:
    """
    Global configuration for the SLH-DAN (Shared-Latent Holistic Dual-Axis Network) pipeline.
    Centralizes hyperparameters, file paths, and model specifications.
    """

    # ==========================
    # General & Reproducibility
    # ==========================
    SEED = 42
    DEBUG = False  # Set to True to use a small subset of data for debugging/testing
    DEBUG_SAMPLE_SIZE = 50  # Number of samples to use if DEBUG is True

    # ==========================
    # Directory Paths
    # ==========================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Cache directory specifically for this solution idea
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_46")

    # Submission directory
    SUBMISSION_DIR = "./submission"

    # Metadata file paths (pre-generated)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Model checkpoint and output paths
    MODEL_SAVE_PATH = os.path.join(CACHE_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # DICOM Source Directories
    # Note: These are relative to INPUT_DIR when constructed
    TRAIN_DICOM_FOLDER = "train"
    TEST_DICOM_FOLDER = "test"

    # ==========================
    # Data Processing (Tri-Slab)
    # ==========================
    IMG_SIZE = 224  # Native resolution for EfficientNet-B0
    SLAB_COUNT = 3  # Number of slabs per view
    SLAB_OVERLAP = 0.15  # 15% overlap between slabs

    # DataLoader settings
    NUM_WORKERS = 8  # Optimized for 12 vCPUs

    # ==========================
    # Model Architecture
    # ==========================
    BACKBONE = "efficientnet_b0"
    BACKBONE_PRETRAINED = True

    # Dimensions
    VISUAL_DIM = 1280  # Output dimension of EfficientNet-B0 GAP (no bottleneck)
    LATENT_DIM = 128  # Dimension of the Shared Latent Tabular Vector

    # ==========================
    # Training Hyperparameters
    # ==========================
    BATCH_SIZE = 32
    EPOCHS = 50
    LR = 1e-4
    WEIGHT_DECAY = 1e-2

    # Early Stopping
    PATIENCE = 8  # Stop if validation score doesn't improve for 8 epochs

    # Scheduler (Cosine Annealing)
    SCHEDULER_T_MAX = 50
    SCHEDULER_MIN_LR = 1e-6

    # ==========================
    # Metric / Loss Constants
    # ==========================
    # Modified Laplace Log Likelihood constants
    METRIC_ERROR_CLIP = 1000.0  # Max error penalty (ml)
    METRIC_CONFIDENCE_CLIP = 70.0  # Min confidence (ml)

    # ==========================
    # Hardware
    # ==========================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """
        Creates necessary working and submission directories.
        Should be called at the start of the pipeline.
        """
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

    @classmethod
    def print_config(cls):
        """
        Prints the current configuration for verification.
        """
        print("\n" + "=" * 40)
        print("CONFIGURATION")
        print("=" * 40)
        for k, v in cls.__dict__.items():
            if not k.startswith("__") and not callable(v):
                print(f"{k}: {v}")
        print("=" * 40 + "\n")
