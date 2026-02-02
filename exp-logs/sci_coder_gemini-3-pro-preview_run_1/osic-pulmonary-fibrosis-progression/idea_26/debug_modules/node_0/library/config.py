import os
import torch


class Config:
    """
    Configuration for Dense-Projection Symmetric Dual-Axis Network (DP-SDAN).
    """

    # ==========================================
    # 1. General Setup
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # ==========================================
    # 2. File Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Metadata files generated in previous steps
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Root directory for DICOM images (metadata contains relative paths from here)
    DICOM_ROOT = INPUT_DIR

    # Cache directory for processed tri-slab images and features
    # Implements the caching requirement for deterministic data processing
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_26")
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Submission output
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # 3. Data Processing
    # ==========================================
    # Image specs for Fixed Overlapping Tri-Slabs
    IMG_SIZE = 224
    SLAB_COUNT = 3
    SLAB_OVERLAP = 0.15  # 15% overlap between slabs

    # Tabular input features: Age(1) + Sex(1) + Percent(1) + SmokingStatus(3) = 6
    TABULAR_INPUT_DIM = 6

    # ==========================================
    # 4. Model Architecture
    # ==========================================
    BACKBONE_NAME = "efficientnet_b0"

    # Native output dimension of EfficientNet-B0 (no bottleneck projection)
    # This preserves high-fidelity visual texture signals.
    EMBED_DIM = 1280

    # Multi-Stage Dense Tabular Expansion
    # Projects 6-dim clinical data to 1280-dim dense vector
    TABULAR_HIDDEN_DIMS = [64, 256, 1280]

    # ==========================================
    # 5. Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 16
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2

    # Early Stopping
    PATIENCE = 8

    # ==========================================
    # 6. Metric Constants (Laplace Log Likelihood)
    # ==========================================
    MAX_ERROR = 1000  # Clip absolute error at 1000 ml
    MIN_SIGMA = 70  # Clip confidence (std dev) at 70 ml

    @classmethod
    def print_config(cls):
        """Prints the current configuration."""
        print("\n" + "=" * 40)
        print("DP-SDAN CONFIGURATION")
        print("=" * 40)
        for k, v in cls.__dict__.items():
            if not k.startswith("__") and not callable(v):
                print(f"{k.ljust(20)}: {v}")
        print("=" * 40 + "\n")
