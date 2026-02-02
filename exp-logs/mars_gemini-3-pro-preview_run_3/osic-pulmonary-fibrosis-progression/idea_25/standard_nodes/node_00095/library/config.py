import os


class Config:
    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    NUM_WORKERS = 4
    DEVICE = "cuda"  # Will be handled by torch.device logic in pipeline, but good to have default

    # =========================================================================
    # Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for caching processed data and saving models
    # Idea 25 corresponds to the current task
    WORKING_DIR = "./working/idea_dsprnet"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")

    # Create directories if they don't exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # Data Preprocessing & Augmentation
    # =========================================================================
    # Image Parameters
    IMG_SIZE = 260  # EfficientNet-B2 native resolution
    SLICES_PER_PATIENT = 3  # Anchor + 2 boundary slices

    # Radiological Windowing (Lung Window)
    DICOM_WINDOW_CENTER = -600
    DICOM_WINDOW_WIDTH = 1500

    # Normalization Constants (Derived from EDA)
    # Target (FVC)
    TARGET_MEAN = 2654.6528
    TARGET_STD = 801.7017

    # Tabular Features
    AGE_MEAN = 67.5825
    AGE_STD = 6.6259
    PERCENT_MEAN = 76.9105
    PERCENT_STD = 19.1970

    # Time Scaling
    TIME_SCALE = 0.01  # Scale weeks by 0.01

    # =========================================================================
    # Model Architecture
    # =========================================================================
    BACKBONE_NAME = "tf_efficientnet_b2_ns"

    # Dimensions
    IMG_EMBED_DIM = 64  # Projection dimension for image features
    CLINICAL_HIDDEN_DIM = 128  # Hidden dim for Stream A (Clinical Anchor)
    LATENT_DIM = 64  # Output dim for Stream A and Stream B
    FUSION_HIDDEN_DIM = 128  # Hidden dim for Stream B (Visual Interaction)

    # Uncertainty
    MIN_UNCERTAINTY = 70.0  # For post-processing clip

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    EPOCHS = 50
    BATCH_SIZE = 16  # Adjusted for 3 slices per patient and B2 backbone memory

    # Optimization
    LR_BACKBONE = 1e-4
    LR_HEAD = 1e-3
    WEIGHT_DECAY = 1e-2

    # Scheduler (Cosine Annealing)
    T_MAX = EPOCHS
    ETA_MIN = 1e-6

    # =========================================================================
    # Loss Function
    # =========================================================================
    # Metric-Aligned Laplace Log Likelihood constants
    # Loss = (sqrt(2) * Delta) / sigma + ln(sqrt(2) * sigma)
    # We define constants here for clarity, though used in loss implementation
    SQRT_2 = 1.41421356
