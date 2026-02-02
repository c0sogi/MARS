import os
import torch


class Config:
    # ==========================================
    # Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Working directory for this specific idea iteration
    WORKING_DIR = "./working/idea_22"

    # Raw Data Files
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata Files (Pre-generated)
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # ==========================================
    # General Setup
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # 12 vCPUs available, 4 is usually a sweet spot for dataloading
    DEBUG = False  # Toggle for debugging on small subsets

    # ==========================================
    # Data Processing
    # ==========================================
    IMAGE_SIZE = 224  # Upsampling to 224x224 via Bicubic
    NUM_CHANNELS = 3  # Band 1 (Norm), Band 2 (Norm), Average
    BATCH_SIZE = 64  # Efficient for ResNet18 on A100

    # ==========================================
    # Model Architecture
    # ==========================================
    BACKBONE = "resnet18"
    PRETRAINED = True
    DROPOUT_RATE = 0.5

    # Angle-Gated Feature Calibration Dimensions
    ANGLE_EMBEDDING_DIM = 64  # Hidden dim for angle MLP
    FEATURE_DIM = 512  # Output dim of ResNet18 GAP

    # ==========================================
    # Training Hyperparameters (Phase 1)
    # ==========================================
    # Optimization (AdamW)
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 0.01
    MAX_EPOCHS = 100  # High ceiling, controlled by Early Stopping

    # Scheduler (ReduceLROnPlateau)
    PATIENCE = 10  # High patience for deep convergence
    FACTOR = 0.5  # Gentle decay
    MIN_LR = 1e-6

    # Loss Function
    LABEL_SMOOTHING = 0.05

    # ==========================================
    # SWA Hyperparameters (Phase 2)
    # ==========================================
    SWA_EPOCHS = 12  # Number of epochs to run SWA
    SWA_LR = 2e-4  # Constant learning rate during SWA

    # ==========================================
    # Evaluation & Ensemble
    # ==========================================
    N_FOLDS = 5  # 5-Fold Stratified CV
    TTA_STEPS = 4  # Klein Four-Group: Original, H-Flip, V-Flip, Rotate180
