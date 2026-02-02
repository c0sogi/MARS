import os
import torch


class Config:
    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    N_FOLDS = 5
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SAMPLE_SIZE = 50
    NUM_WORKERS = 4  # Optimized for 12 vCPUs

    # =========================================================================
    # Directories & Paths
    # =========================================================================
    INPUT_DIR = "./input"
    IMAGES_DIR = os.path.join(INPUT_DIR, "images")
    METADATA_DIR = "./metadata"

    # Metadata files (Pre-generated)
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output Directories
    # Using 'idea_7' as the working directory for this iteration
    WORKING_DIR = "./working/idea_7"
    SUBMISSION_FILE = "./submission.csv"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # =========================================================================
    # Model Architectures (Heterogeneous Ensemble)
    # =========================================================================
    # Model A: EfficientNetV2-L
    # High capacity, compound scaling, Fused-MBConv
    MODEL_A_NAME = "tf_efficientnetv2_l.in21k_ft_in1k"
    IMG_SIZE_EFFNET = 480  # Native resolution for Model A

    # Model B: ConvNeXt-Base
    # Modernized ResNet with Transformer-style blocks
    MODEL_B_NAME = "convnext_base.fb_in22k_ft_in1k"
    IMG_SIZE_CONVNEXT = 384  # Native resolution for Model B

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 16
    EPOCHS = 15
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2
    MAX_GRAD_NORM = 10.0

    # Gradient Accumulation & AMP
    ACCUMULATION_STEPS = 1
    USE_AMP = True

    # Regularization
    LABEL_SMOOTHING = 0.05

    # Multi-Sample Dropout (MSD) Settings
    # Applies multiple dropout masks to the final layer to accelerate convergence
    MSD_NUM_DROPOUTS = 5
    MSD_DROPOUT_RATE = 0.5

    # =========================================================================
    # Data & Augmentation
    # =========================================================================
    # CoarseDropout settings
    AUG_HOLES_NUM_MIN = 2
    AUG_HOLES_NUM_MAX = 8
    AUG_HOLE_SIZE_MIN = 16
    AUG_HOLE_SIZE_MAX_EFFNET = 100  # Scaled for 480px
    AUG_HOLE_SIZE_MAX_CONVNEXT = 80  # Scaled for 384px

    # =========================================================================
    # Targets & Inference
    # =========================================================================
    # We decompose the 4-class problem into 2 binary attributes
    TARGET_COLS = ["rust", "scab"]

    # Mapping for final reconstruction
    # Healthy = (1-r)*(1-s), Rust = r*(1-s), Scab = (1-r)*s, Multiple = r*s
    CLASS_LABELS = ["healthy", "multiple_diseases", "rust", "scab"]

    # =========================================================================
    # Hardware
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def print_config(cls):
        print(f"{'='*30}")
        print(f"CONFIGURATION")
        print(f"{'='*30}")
        print(f"Device: {cls.DEVICE}")
        print(f"Model A: {cls.MODEL_A_NAME} ({cls.IMG_SIZE_EFFNET}px)")
        print(f"Model B: {cls.MODEL_B_NAME} ({cls.IMG_SIZE_CONVNEXT}px)")
        print(f"Folds: {cls.N_FOLDS}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Epochs: {cls.EPOCHS}")
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        print(f"Working Dir: {cls.WORKING_DIR}")
        print(f"{'='*30}")
