import os
import torch


class Config:
    # ==========================================
    # 1. Paths and Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Cache directory for processed images (Idea 70 specific)
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_70")

    # Metadata file paths
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # DICOM directories
    DICOM_DIR_TRAIN = os.path.join(INPUT_DIR, "train")
    DICOM_DIR_TEST = os.path.join(INPUT_DIR, "test")

    # Submission path
    SUBMISSION_FILE = os.path.join(WORKING_DIR, "submission.csv")

    # ==========================================
    # 2. Data Preprocessing & Augmentation
    # ==========================================
    # Fixed Overlapping Orthogonal Tri-Slabs
    IMG_SIZE = 224  # Native resolution for EfficientNet-B0
    N_SLABS = 3  # 3 slabs -> RGB channels
    SLAB_OVERLAP = 0.15  # 15% overlap between slabs

    # Views to extract
    VIEWS = ["axial", "coronal"]

    # Tabular Features used in Shared Latent Encoder
    TABULAR_FEATURES = ["Age", "Sex", "SmokingStatus", "Percent"]

    # Normalization constants (ImageNet defaults)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # ==========================================
    # 3. Model Architecture (PGBB-Net)
    # ==========================================
    BACKBONE_NAME = "efficientnet_b0"
    BACKBONE_PRETRAINED = True

    # Dimensions
    BACKBONE_OUT_DIM = 1280  # Native B0 output (Global Average Pooling)
    LATENT_DIM = 128  # Shared Latent Vector (T_lat)
    BOTTLENECK_DIM = 128  # Balanced Bottleneck

    # ==========================================
    # 4. Training Hyperparameters
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Optimization
    BATCH_SIZE = 32
    EPOCHS = 50
    LR = 1e-4
    WEIGHT_DECAY = 1e-2

    # Scheduler (Cosine Annealing)
    T_MAX = EPOCHS
    ETA_MIN = 1e-6

    # Early Stopping
    PATIENCE = 8

    # DataLoader
    NUM_WORKERS = 4

    # ==========================================
    # 5. Metric & Loss Constraints
    # ==========================================
    # Modified Laplace Log Likelihood constants
    ERROR_CLIP = 1000.0  # Max absolute error penalty
    CONFIDENCE_CLIP = 70.0  # Min confidence (sigma)

    # ==========================================
    # 6. Debugging / Development
    # ==========================================
    # Set to True to run on a small subset of data
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 20
