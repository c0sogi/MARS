import os
import torch


class Config:
    # ==========================================
    # Project & Experiment Setup
    # ==========================================
    PROJECT_NAME = "Cactus_Identification"
    IDEA_NAME = "idea_5"

    # ==========================================
    # Directories & Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for this specific idea
    WORKING_DIR = os.path.join("./working", IDEA_NAME)
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")

    # Ensure working directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output Paths
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Configuration
    # ==========================================
    IMAGE_SIZE = (32, 32)
    NUM_CLASSES = 1
    BATCH_SIZE = 128
    NUM_WORKERS = 4

    # Debugging / Quick Run
    # Set DEBUG to True to run on a small subset of data for testing pipeline
    DEBUG = False
    DEBUG_SAMPLES = 500

    # ==========================================
    # Model Architecture (Custom SE-ResNet)
    # ==========================================
    MODEL_NAME = "CustomSEResNet"

    # Hyperparameters for the lightweight architecture
    # Designed to maintain 8x8 feature map at the end
    MODEL_PARAMS = {
        "in_channels": 3,
        "num_classes": 1,
        "base_channels": 64,
        "layers": [2, 2, 2],  # Depth of residual blocks per stage
        "strides": [1, 2, 2],  # Downsampling schedule (32->32->16->8)
        "se_reduction": 16,  # Squeeze-and-Excitation reduction ratio
        "dropout": 0.0,  # Minimal regularization for ResNet
    }

    # ==========================================
    # Training Configuration
    # ==========================================
    # Homogeneous Seed Averaging
    SEEDS = [0, 1, 2, 3, 4]

    EPOCHS = 30
    EARLY_STOPPING_PATIENCE = 6

    # Optimization
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2
    OPTIMIZER_NAME = "AdamW"

    # Scheduler
    SCHEDULER_NAME = "CosineAnnealingLR"
    MIN_LR = 1e-6

    # Loss
    LOSS_FN = "BCEWithLogitsLoss"

    # ==========================================
    # Inference Configuration
    # ==========================================
    USE_TTA = True  # Test Time Augmentation

    # ==========================================
    # Hardware
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def get_model_path(cls, seed):
        """Returns the file path for saving/loading a model checkpoint for a specific seed."""
        return os.path.join(cls.WORKING_DIR, f"model_seed_{seed}.pth")
