import os
import torch


class Config:
    # ==========================================
    # System & Hardware
    # ==========================================
    PROJECT_NAME = "Cactus_Identification_RepVGG"
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Using 4 workers as a safe default for 12 vCPUs
    NUM_WORKERS = 4

    # ==========================================
    # File Paths
    # ==========================================
    # Input Directories
    INPUT_DIR = "./input"
    TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "train")
    TEST_IMG_DIR = os.path.join(INPUT_DIR, "test")

    # Metadata
    METADATA_DIR = "./metadata"
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Working Directory for Idea 5 (Caching & Checkpoints)
    WORKING_DIR = "./working/idea_5"
    CHECKPOINT_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure working and submission directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Data Parameters
    # ==========================================
    IMG_SIZE = 32
    NUM_CLASSES = 1

    # Normalization constants derived from Data Analysis:
    # Mean: R=128.37, G=115.25, B=119.40 -> /255 -> [0.5034, 0.4520, 0.4682]
    # Std : R=38.60,  G=35.68,  B=39.15  -> /255 -> [0.1514, 0.1399, 0.1535]
    NORM_MEAN = [0.5034, 0.4520, 0.4682]
    NORM_STD = [0.1514, 0.1399, 0.1535]

    # ==========================================
    # Model Architecture (RepVGG-style)
    # ==========================================
    # Channel widths for stages.
    # Conservative stem -> Stage 1 -> Stage 2 -> Stage 3
    # Keeps capacity high for small images.
    MODEL_CHANNELS = [64, 128, 256, 512]

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    EPOCHS = 30
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2  # For AdamW

    # Mixup Regularization
    MIXUP_ALPHA = 0.2

    # Scheduler
    ETA_MIN = 1e-6  # Minimum LR for Cosine Annealing

    # Early Stopping
    PATIENCE = 10  # Stop if no improvement for 10 epochs

    # ==========================================
    # Inference
    # ==========================================
    # Test Time Augmentation
    USE_TTA = True

    def __init__(self):
        # Print config summary on initialization
        print(f"Config initialized for {self.PROJECT_NAME}")
        print(f"Device: {self.DEVICE}")
        print(f"Working Dir: {self.WORKING_DIR}")
