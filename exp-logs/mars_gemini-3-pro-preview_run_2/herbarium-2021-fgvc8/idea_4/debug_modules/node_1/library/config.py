import os
import torch


class Config:
    # ==========================================
    # Path Configuration
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_4"
    SUBMISSION_DIR = "./submission"

    # Input Metadata Files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    TRAIN_METADATA_JSON = os.path.join(INPUT_DIR, "train/metadata.json")

    # Cache and Output Files
    # Caching taxonomy mappings to avoid re-parsing large JSONs
    TAXONOMY_MAP_PATH = os.path.join(WORKING_DIR, "taxonomy_mappings.parquet")
    MODEL_CHECKPOINT = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Configuration
    # ==========================================
    # Image resolution set to 224x224 for throughput
    IMAGE_SIZE = 224

    # Number of species classes
    NUM_CLASSES = 64500

    # Compute settings
    NUM_WORKERS = 12
    BATCH_SIZE = 128  # Optimized for A100 40GB

    # Debugging flags
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 5000  # Number of samples to use if DEBUG is True

    # ==========================================
    # Model Configuration
    # ==========================================
    BACKBONE = "convnext_tiny"
    PRETRAINED = True
    HEAD_HIDDEN_DIM = 512

    # ==========================================
    # Training Configuration
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Stage 1: Representation Learning
    # Instance-balanced sampling, training backbone + heads
    STAGE1_EPOCHS = 4
    STAGE1_LR = 1e-3
    STAGE1_WEIGHT_DECAY = 1e-4
    LABEL_SMOOTHING = 0.1

    # Stage 2: Classifier Re-balancing
    # Class-balanced sampling, frozen backbone, fine-tune heads
    STAGE2_EPOCHS = 1
    STAGE2_LR = 1e-4

    @classmethod
    def setup(cls):
        """
        Creates necessary working and submission directories.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

    @classmethod
    def print_config(cls):
        """
        Prints the current configuration.
        """
        print("=" * 30)
        print("CONFIGURATION")
        print("=" * 30)
        for k, v in cls.__dict__.items():
            if not k.startswith("__") and not callable(v):
                print(f"{k}: {v}")
        print("=" * 30)
