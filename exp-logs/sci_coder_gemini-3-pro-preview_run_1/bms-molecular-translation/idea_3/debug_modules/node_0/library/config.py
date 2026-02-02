import os
import torch


class Config:
    """
    Configuration class for the Visual Transformer (CNN-Transformer) InChI predictor.
    """

    # ==========================================
    # General Settings
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to use a small subset of data for debugging
    NUM_WORKERS = 4  # Number of data loading workers

    # ==========================================
    # Directories and Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_3"
    SUBMISSION_DIR = "./submission"

    # Input Files (using generated metadata)
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output Files
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    TOKENIZER_PATH = os.path.join(WORKING_DIR, "tokenizer.npy")
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Hyperparameters
    # ==========================================
    IMAGE_SIZE = 224  # Input image resolution (224x224)
    # Max length observed in EDA was 403.
    # We add a buffer for <SOS> and <EOS> tokens.
    MAX_LEN = 410

    # ==========================================
    # Model Architecture
    # ==========================================
    # Encoder
    ENCODER_NAME = "resnet18"  # Lightweight CNN backbone
    ENCODER_PRETRAINED = True

    # Decoder (Transformer)
    D_MODEL = 256  # Embedding dimension
    N_HEAD = 4  # Number of attention heads
    N_LAYER = 3  # Number of decoder layers
    FF_DIM = 512  # Feed-forward network dimension
    DROPOUT = 0.1  # Dropout rate

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 128  # A100 has 40GB RAM, can handle larger batches
    EPOCHS = 15  # Total training epochs
    LEARNING_RATE = 1e-4  # Initial learning rate
    WEIGHT_DECAY = 1e-6  # Weight decay for AdamW
    MAX_GRAD_NORM = 5.0  # Gradient clipping

    # Early Stopping
    PATIENCE = 3  # Epochs to wait before stopping if val_loss doesn't improve

    # Scheduler
    T_MAX = 15  # For CosineAnnealingLR (usually same as EPOCHS)
    MIN_LR = 1e-6  # Minimum learning rate

    # ==========================================
    # Compute
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """
        Creates necessary output directories if they don't exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        print(f"Config setup complete. Working dir: {cls.WORKING_DIR}")
        print(f"Device: {cls.DEVICE}")
