import os
import torch


class Config:
    # ==========================================
    # 1. General Configuration
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # ==========================================
    # 2. File Paths & Directories
    # ==========================================
    # Root directories
    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Specific experiment cache directory (Idea 15)
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_15")

    # Checkpoint and Submission directories
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"

    # Input Metadata Files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_ROOT, "sample_submission.csv")

    # Output Files
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model_idea_15.pth")
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Create directories if they don't exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # 3. Data Processing Hyperparameters
    # ==========================================
    IMG_SIZE = 224

    # Tri-Slab Generation
    N_SLABS = 3
    SLAB_OVERLAP = 0.15

    # Tabular Features to Tokenize
    TABULAR_FEATURES = ["Age", "Sex", "SmokingStatus", "Percent"]

    # Normalization Constants (Approximate from EDA)
    AGE_MEAN = 67.0
    AGE_STD = 7.0
    PERCENT_MEAN = 77.0
    PERCENT_STD = 20.0

    # ==========================================
    # 4. Model Architecture Hyperparameters
    # ==========================================
    # Visual Backbone
    BACKBONE_NAME = "efficientnet_b0"
    PRETRAINED = True
    VISUAL_DIM = 1280  # Output dim of EfficientNet-B0 GAP

    # Granular Tabular Tokenization
    # We project all tabular tokens to the visual dimension for symmetric attention
    TOKEN_DIM = 1280

    # Fusion Transformer
    NUM_ATTENTION_HEADS = 8
    NUM_ATTENTION_LAYERS = 2
    DROPOUT = 0.1

    # Parametric Head
    # Outputs: alpha (slope), sigma_base, sigma_growth
    OUTPUT_DIM = 3

    # ==========================================
    # 5. Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 32
    N_EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2

    # Scheduler
    T_MAX = 50  # For CosineAnnealingLR
    ETA_MIN = 1e-6

    # Early Stopping
    PATIENCE = 8

    # Metric / Loss Constants
    Q_CLIP = 70  # Confidence clipping (sigma_clipped)
    MAX_ERR = 1000  # Error thresholding (Delta)

    # Debugging limits
    DEBUG_SAMPLES = 50 if DEBUG else None
