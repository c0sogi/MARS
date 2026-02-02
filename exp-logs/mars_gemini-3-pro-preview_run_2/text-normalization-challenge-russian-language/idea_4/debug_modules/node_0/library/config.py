import os
import torch


class Config:
    # ==========================================
    # 1. Environment & Hardware
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # ==========================================
    # 2. File Paths
    # ==========================================
    # Input Data (Metadata)
    TRAIN_DATA = "./metadata/train.csv"
    VAL_DATA = "./metadata/val.csv"
    TEST_DATA = "./metadata/test.csv"

    # Output & Working Directories
    WORK_DIR = "./working/idea_4/"
    SUBMISSION_PATH = "./submission/submission.csv"

    # Caching Directories
    HFBB_CACHE_DIR = os.path.join(WORK_DIR, "hfbb_cache")
    TRANSFORMER_CACHE_DIR = os.path.join(WORK_DIR, "transformer_cache")

    # Model Checkpoints & Vocab
    MODEL_CHECKPOINT = os.path.join(WORK_DIR, "transformer_best.pth")
    VOCAB_PATH = os.path.join(WORK_DIR, "vocab.json")

    # ==========================================
    # 3. HFBB (Tier 1) Configuration
    # ==========================================
    # Order of lookup for the memory engine
    HFBB_ORDER = ["trigram", "bigram_prev", "bigram_next", "unigram"]

    # ==========================================
    # 4. Transformer (Tier 2) Configuration
    # ==========================================
    # Gating Mechanism: Regex to trigger Tier 2 if Tier 1 fails
    # Matches tokens with digits or latin characters (for transliteration)
    GATE_REGEX = r"(\d|[a-zA-Z])"

    # Data Processing
    CONTEXT_WINDOW = 1  # Number of tokens to include on left/right
    MAX_SEQ_LEN = 128  # Max length for character sequences

    # Class Balancing / Upsampling
    # We upsample these rare classes to match the frequency of the reference class
    UPSAMPLE_CLASSES = [
        "MONEY",
        "DECIMAL",
        "TELEPHONE",
        "ELECTRONIC",
        "DIGIT",
        "TIME",
        "MEASURE",
        "ORDINAL",
        "FRACTION",
    ]
    REFERENCE_CLASS_FOR_UPSAMPLING = "DATE"

    # Model Architecture (Character-level Transformer)
    D_MODEL = 256
    NHEAD = 4
    NUM_ENCODER_LAYERS = 4
    NUM_DECODER_LAYERS = 4
    DIM_FEEDFORWARD = 1024
    DROPOUT = 0.2

    # ==========================================
    # 5. Training Hyperparameters
    # ==========================================
    DEBUG = False  # Set to True to use a small subset of data
    DEBUG_SIZE = 10000  # Number of samples if DEBUG is True

    BATCH_SIZE = 128
    LEARNING_RATE = 3e-4
    NUM_EPOCHS = 15
    EARLY_STOPPING_PATIENCE = 3
    WARMUP_STEPS = 1000
    WEIGHT_DECAY = 0.01

    @classmethod
    def setup(cls):
        """Creates necessary directories and sets deterministic flags."""
        os.makedirs(cls.WORK_DIR, exist_ok=True)
        os.makedirs(cls.HFBB_CACHE_DIR, exist_ok=True)
        os.makedirs(cls.TRANSFORMER_CACHE_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(cls.SUBMISSION_PATH), exist_ok=True)

        # Set deterministic behavior for PyTorch
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
