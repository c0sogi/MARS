import os
import torch


class Config:
    # ==========================================
    # 1. Environment & Paths
    # ==========================================
    SEED = 42

    # Input Paths
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "train")
    TEST_IMG_DIR = os.path.join(INPUT_DIR, "test")

    # Metadata Files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Paths
    # We use a specific subdirectory for this idea to avoid conflicts
    IDEA_ID = "idea_73"
    WORKING_DIR = os.path.join("./working", IDEA_ID)
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Create directories
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # 2. Data Preprocessing & Augmentation
    # ==========================================
    # Image Generation
    IMG_SIZE = 224  # Native resolution for EfficientNet-B0
    SLAB_COUNT = 3  # Tri-slab generation
    OVERLAP_RATIO = 0.15  # 15% overlap between slabs

    # Views to generate
    VIEWS = ["axial", "coronal"]

    # Normalization (ImageNet stats)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # DataLoader
    BATCH_SIZE = 16  # Conservative batch size for dual-backbone
    NUM_WORKERS = 4
    PIN_MEMORY = True

    # ==========================================
    # 3. Model Architecture (PCCG-Net)
    # ==========================================
    # Visual Backbone
    BACKBONE_NAME = "efficientnet_b0"
    BACKBONE_PRETRAINED = True
    BACKBONE_OUT_DIM = 1280  # Native output dim of B0 (no projection)

    # Tabular / Latent Topology
    TABULAR_INPUT_DIM = 7  # Age, Sex, Smoking, Percent + OneHot encodings
    LATENT_DIM = 128  # Shared Latent Vector (T_lat)

    # Fusion & Context
    ALIGN_DIM = 1280  # Alignment dimension for tabular token (matches backbone)
    CONTEXT_DIM = 128  # Bottleneck dimension after fusion (H_ctx)

    # Attention
    NUM_HEADS = 4
    ATTN_DROPOUT = 0.1
    FFN_DIM = 512

    # General
    DROPOUT_RATE = 0.2

    # ==========================================
    # 4. Training Hyperparameters
    # ==========================================
    EPOCHS = 50
    PATIENCE = 8  # Strict patience for Early Stopping

    # Optimizer (AdamW)
    LR = 1e-4
    WEIGHT_DECAY = 1e-2

    # Scheduler (Cosine Annealing)
    T_MAX = 50
    ETA_MIN = 1e-6

    # Gradient Clipping
    CLIP_GRAD_NORM = 1.0

    # ==========================================
    # 5. Metric & Loss Constants
    # ==========================================
    # Modified Laplace Log Likelihood
    MAX_ERROR = 1000.0  # Clip absolute error
    MIN_CONFIDENCE = 70.0  # Clip confidence (sigma)

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def print_config(cls):
        print("=" * 40)
        print(f"CONFIG: {cls.IDEA_ID}")
        print("=" * 40)
        print(f"Device: {cls.DEVICE}")
        print(f"Backbone: {cls.BACKBONE_NAME} (Dim: {cls.BACKBONE_OUT_DIM})")
        print(f"Resolution: {cls.IMG_SIZE}x{cls.IMG_SIZE}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Latent Dim: {cls.LATENT_DIM}")
        print(f"Patience: {cls.PATIENCE}")
        print(f"Cache Dir: {cls.CACHE_DIR}")
        print("=" * 40)
