import os
import torch


class Config:
    """
    Configuration parameters for the Spatial Attention-Guided Recurrent Network pipeline.
    """

    # ---------------------------------------------------------
    # General Settings
    # ---------------------------------------------------------
    SEED = 42
    DEBUG = False  # Set to True to use a smaller subset of data for debugging
    DEBUG_SAMPLE_SIZE = 1000  # Number of samples to use in debug mode

    # ---------------------------------------------------------
    # Directories & Paths
    # ---------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_4"
    SUBMISSION_DIR = "./submission"

    # Input Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Files
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    TOKENIZER_PATH = os.path.join(WORKING_DIR, "tokenizer.json")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ---------------------------------------------------------
    # Data Parameters
    # ---------------------------------------------------------
    IMAGE_SIZE = 320  # Input image size (square)
    IMAGE_CHANNELS = 1  # Grayscale

    # Sequence generation
    # Max length observed in EDA was 403. Adding buffer for SOS, EOS, and margin.
    MAX_SEQUENCE_LENGTH = 450

    # Special Tokens
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"

    # ---------------------------------------------------------
    # Model Architecture Hyperparameters
    # ---------------------------------------------------------
    # Encoder
    ENCODER_NAME = "mobilenet_v3_small"  # Lightweight CNN backbone
    ENCODER_DIM = 256  # Dimension of the feature grid after 1x1 conv

    # Decoder
    EMBED_DIM = 256  # Character embedding dimension
    DECODER_DIM = 512  # Hidden state dimension of the GRU
    ATTENTION_DIM = 256  # Dimension for Bahdanau Attention internal layer
    DROPOUT = 0.5  # Dropout rate

    # ---------------------------------------------------------
    # Training Hyperparameters
    # ---------------------------------------------------------
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-6
    EPOCHS = 15
    NUM_WORKERS = 4  # Number of dataloader workers

    # Training Strategy
    TEACHER_FORCING_RATIO = 0.5  # Probability of using ground truth as next input
    CLIP_GRAD = 5.0  # Gradient clipping max norm
    EARLY_STOPPING_PATIENCE = (
        3  # Epochs to wait before stopping if val loss doesn't improve
    )

    # ---------------------------------------------------------
    # Compute
    # ---------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup_directories(cls):
        """
        Ensure working and submission directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        print(f"Directories ensured: {cls.WORKING_DIR}, {cls.SUBMISSION_DIR}")


# Automatically setup directories when config is imported/used
Config.setup_directories()
