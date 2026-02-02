import os
import torch


class Config:
    # -------------------------------------------------------------------------
    # Directories & Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_2"
    OUTPUT_DIR = "./submission"

    # Metadata Files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Raw metadata for taxonomy extraction
    RAW_TRAIN_METADATA = os.path.join(INPUT_DIR, "train", "metadata.json")

    # Cache and Output Files
    TAXONOMY_MAP_PATH = os.path.join(WORKING_DIR, "taxonomy_mappings.parquet")
    MODEL_CHECKPOINT_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_FILE = os.path.join(OUTPUT_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Data Configuration
    # -------------------------------------------------------------------------
    IMAGE_SIZE = 224
    NUM_CLASSES = 64500  # Based on data analysis
    NUM_WORKERS = 12

    # -------------------------------------------------------------------------
    # Model Configuration
    # -------------------------------------------------------------------------
    BACKBONE = "efficientnet_b0"
    EMBEDDING_DIM = 512
    DROPOUT = 0.2

    # ArcFace Head Parameters (Metric Learning)
    ARCFACE_SCALE = 30.0
    ARCFACE_MARGIN = 0.50

    # -------------------------------------------------------------------------
    # Training Configuration
    # -------------------------------------------------------------------------
    SEED = 42
    BATCH_SIZE = 256  # Optimized for A100 40GB VRAM
    EPOCHS = 12  # Targeted for 24h runtime with full dataset
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    MAX_GRAD_NORM = 10.0

    # Loss Weights for Hierarchical Multi-Task Learning
    # Total Loss = Species_Loss + alpha * Family_Loss + beta * Order_Loss
    LOSS_WEIGHT_SPECIES = 1.0
    LOSS_WEIGHT_FAMILY = 0.1
    LOSS_WEIGHT_ORDER = 0.1

    # Regularization
    LABEL_SMOOTHING = 0.1

    # -------------------------------------------------------------------------
    # Hardware & Runtime
    # -------------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------------------------------------------------------
    # Debugging / Development
    # -------------------------------------------------------------------------
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 5000

    @classmethod
    def setup(cls):
        """
        Initialize the working environment by creating necessary directories.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
