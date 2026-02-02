import os
import torch


class Config:
    # ==========================================
    # System & Paths
    # ==========================================
    PROJECT_NAME = "contrails_identification"
    IDEA_NAME = "idea_10"  # Cascaded ResNet18 U-Net

    # Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = os.path.join("./working", IDEA_NAME)
    SUBMISSION_DIR = "./submission"

    # Ensure working and submission directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "validation_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Model Checkpoint
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # ==========================================
    # Data Parameters
    # ==========================================
    SEED = 42
    IMG_SIZE = 256

    # Temporal sequence details
    N_TIMES_BEFORE = 4
    N_TIMES_AFTER = 3
    TOTAL_FRAMES = N_TIMES_BEFORE + N_TIMES_AFTER + 1  # 8 frames

    # Input Engineering
    # Stage 1 inputs: 3 channels (Ash FCC) + 3 channels (Temporal Diff) = 6 channels
    IN_CHANNELS_STAGE1 = 6
    # Stage 2 inputs: 6 channels (Original) + 1 channel (Stage 1 Prob) = 7 channels
    IN_CHANNELS_STAGE2 = 7

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 1e-2  # For AdamW

    # Scheduler
    T_MAX = EPOCHS  # For CosineAnnealingLR
    ETA_MIN = 1e-6

    # Loss Weights
    # Total Loss = Loss_Stage1 + Loss_Stage2
    # Both stages use Hybrid Loss (BCE + BatchDice)

    # ==========================================
    # Model Architecture
    # ==========================================
    BACKBONE = "resnet18"
    ENCODER_WEIGHTS = "imagenet"

    # ==========================================
    # Compute
    # ==========================================
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # Inference & Post-processing
    # ==========================================
    THRESHOLD = 0.5
    USE_TTA = True  # Test-Time Augmentation

    # Debugging
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 100  # If DEBUG is True, limit dataset size
