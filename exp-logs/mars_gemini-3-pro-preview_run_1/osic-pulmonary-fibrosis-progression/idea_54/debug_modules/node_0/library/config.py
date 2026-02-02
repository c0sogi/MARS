import os
import torch


class Config:
    # ==========================================
    # 1. File Paths & Directories
    # ==========================================
    # Root directory for read-only input data
    INPUT_ROOT = "./input"

    # Directory containing the generated metadata CSVs
    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_ROOT, "sample_submission.csv")

    # Working directory for this specific idea (caching, checkpoints)
    # Using 'idea_54' as the identifier for this iteration
    WORKING_DIR = "./working/idea_54"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"

    # Ensure working directories exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # 2. Data Preprocessing & Augmentation
    # ==========================================
    # Image Resolution: 224x224 matches EfficientNet-B0 native resolution
    IMG_SIZE = 224

    # Tri-Slab Generation
    SLAB_COUNT = 3
    SLAB_OVERLAP = 0.15  # 15% overlap between slabs

    # Normalization constants (ImageNet defaults)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # DataLoader settings
    BATCH_SIZE = 32
    NUM_WORKERS = 4  # 12 vCPUs available

    # ==========================================
    # 3. Model Architecture (NSL-HN)
    # ==========================================
    BACKBONE_NAME = "efficientnet_b0"
    PRETRAINED = True

    # Dimensions
    VISUAL_DIM = 1280  # EfficientNet-B0 output channels
    TABULAR_LATENT_DIM = 128  # Shared latent vector size
    FUSED_DIM = 1280  # Dimension after projection/fusion

    # Tabular features used
    TABULAR_COLS = ["Weeks", "Percent", "Age", "Sex", "SmokingStatus"]

    # ==========================================
    # 4. Training Hyperparameters
    # ==========================================
    SEED = 42
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2

    # Scheduler (Cosine Annealing)
    T_MAX = EPOCHS
    ETA_MIN = 1e-6

    # Early Stopping
    PATIENCE = 8

    # Metric Configuration
    MAX_ERROR = 1000  # Clip absolute error
    CONFIDENCE_CLIP = 70  # Clip predicted sigma

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def print_config():
        """Prints the current configuration."""
        print("=" * 40)
        print("CONFIG")
        print("=" * 40)
        for attr in dir(Config):
            if not attr.startswith("__") and not callable(getattr(Config, attr)):
                print(f"{attr}: {getattr(Config, attr)}")
        print("=" * 40)
