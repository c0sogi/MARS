import os


class Config:
    # ==========================================
    # Directories and Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_3"
    SUBMISSION_DIR = "./submission"

    # Ensure output directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata Files
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Vocabulary File
    VOCAB_PATH = os.path.join(WORKING_DIR, "vocab.json")

    # Checkpoints
    CHECKPOINT_PATH = os.path.join(WORKING_DIR, "checkpoint.pth")
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "model_best.pth")
    PREDICTIONS_CSV = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Configuration
    # ==========================================
    # Image parameters
    IMAGE_SIZE = 256
    PATCH_SIZE = 16
    IN_CHANNELS = 1  # EDA indicated images are grayscale

    # Derived parameters
    NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2  # 256 (16*16 grid)

    # Text parameters
    # Max length observed in EDA was 403.
    # We add margin for SOS, EOS and potential longer sequences in test.
    MAX_TEXT_LEN = 512

    # Special Tokens
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    # Decoder-Only Transformer dimensions
    D_MODEL = 384  # Embedding dimension
    N_LAYERS = 6  # Number of transformer blocks
    N_HEADS = 12  # Number of attention heads (384 / 12 = 32 dim per head)
    D_FF = 1536  # Feed-forward dimension (usually 4 * D_MODEL)
    DROPOUT = 0.1  # Dropout rate

    # ==========================================
    # Training Configuration
    # ==========================================
    BATCH_SIZE = 64
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 1e-2
    EPOCHS = 15
    NUM_WORKERS = 4  # Number of dataloader workers
    SEED = 42  # Random seed for reproducibility

    # Learning Rate Scheduler
    WARMUP_STEPS = 1000

    # Early Stopping
    PATIENCE = 3  # Stop after N epochs without validation improvement

    # ==========================================
    # Debug / Development
    # ==========================================
    # If True, runs on a small subset of data for quick verification
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 2000
