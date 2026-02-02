import os


class Config:
    # -------------------------------------------------------------------------
    # Directory and File Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test")

    METADATA_DIR = "./metadata"
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    WORKING_DIR = "./working/idea_15"
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure working directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Data Hyperparameters
    # -------------------------------------------------------------------------
    IMAGE_SIZE = (32, 32)
    NUM_CLASSES = 1

    # Augmentation: Only H-Flip and V-Flip as per strategy
    AUG_H_FLIP_PROB = 0.5
    AUG_V_FLIP_PROB = 0.5

    # -------------------------------------------------------------------------
    # Model Architecture Hyperparameters
    # -------------------------------------------------------------------------
    # Custom Narrow SE-ResNet
    # Channels for the 3 stages.
    # Stage 1: 32x32, Stage 2: 16x16, Stage 3: 8x8
    BACKBONE_CHANNELS = [16, 32, 64]

    # Enable Squeeze-and-Excitation blocks
    USE_SE = True

    # Hybrid Multi-Order Multi-Scale Pooling
    # Extract features from Stage 2 (16x16, 32ch) and Stage 3 (8x8, 64ch)
    # Indices correspond to BACKBONE_CHANNELS list
    POOLING_STAGES_INDICES = [1, 2]

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    # Homogeneous Seed Averaging
    SEEDS = [0, 1, 2, 3, 4]

    # Training Schedule
    EPOCHS = 15
    BATCH_SIZE = 128

    # Optimizer (AdamW)
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2

    # Scheduler (Cosine Annealing)
    # T_max usually equals total epochs
    SCHEDULER_T_MAX = EPOCHS

    # Hardware
    NUM_WORKERS = 4

    # -------------------------------------------------------------------------
    # Inference Hyperparameters
    # -------------------------------------------------------------------------
    # Test Time Augmentation
    USE_TTA = True
    # TTA variants to average over (in addition to original)
    TTA_FLIPS = ["horizontal", "vertical"]
