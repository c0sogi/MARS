import os
import torch
import random
import numpy as np


class Config:
    # --------------------------------------------------------------------------
    # General & Reproducibility
    # --------------------------------------------------------------------------
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging

    # --------------------------------------------------------------------------
    # Directories & Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_7"
    SUBMISSION_DIR = "./submission"

    # Create necessary directories
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata Files (Pre-generated)
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Artifacts (Saved/Loaded from Working Dir)
    TOKENIZER_PATH = os.path.join(WORKING_DIR, "tokenizer.json")
    ATTR_STATS_PATH = os.path.join(WORKING_DIR, "attr_stats.npy")
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Data Configuration
    # --------------------------------------------------------------------------
    IMAGE_SIZE = (256, 256)

    # EDA showed max length is 403. We use 410 to be safe and accommodate special tokens.
    MAX_LEN = 410

    # Attributes for the auxiliary branch:
    # We track counts of Carbon, Hydrogen, Oxygen, Nitrogen, Sulfur,
    # sum of Halogens (F, Cl, Br, I), and the total string Length.
    ATTR_COLS = ["C", "H", "O", "N", "S", "Halogen", "Length"]
    NUM_ATTRIBUTES = len(ATTR_COLS)

    # --------------------------------------------------------------------------
    # Model Architecture (AM-ViT)
    # --------------------------------------------------------------------------
    ENCODER_NAME = "efficientnet_b0"
    # EfficientNet-B0 outputs 1280 channels at the final layer
    ENCODER_DIM = 1280

    # Transformer Decoder settings
    EMBED_DIM = 256
    DECODER_LAYERS = 4
    DECODER_HEADS = 8
    DECODER_FF_DIM = 1024
    DROPOUT = 0.1

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    BATCH_SIZE = 64
    NUM_EPOCHS = 15
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2

    # Early Stopping
    PATIENCE = 3

    # Multi-task Loss Balancing
    # Weight for the attribute regression loss (MSE) relative to the captioning loss (CE)
    # Since attributes are Z-score normalized, a weight of 1.0 is a good starting point.
    ATTR_LOSS_WEIGHT = 1.0

    # System
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed=42):
    """
    Sets the seed for generating random numbers to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Set seed immediately upon import
seed_everything(Config.SEED)
