import os
import torch


class Config:
    """
    Configuration for the Residual-Optimized Hybrid Cascade Text Normalization system.
    """

    # ==========================================
    # 1. Environment & Paths
    # ==========================================
    SEED = 42

    # Directory Structure
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_6"
    SUBMISSION_DIR = "./submission"

    # Input Files (from Metadata)
    TRAIN_FILE = os.path.join(METADATA_DIR, "train.csv")
    VAL_FILE = os.path.join(METADATA_DIR, "val.csv")
    TEST_FILE = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "ru_sample_submission_2.csv")

    # Caching & Intermediate Files
    # HFBB (Tier 1) Cache Directory
    HFBB_CACHE_DIR = os.path.join(WORKING_DIR, "hfbb_cache")

    # Residual Dataset Cache (Tier 2 Training Data)
    # Stores the "hard" examples generated via K-Fold Jackknifing
    RESIDUAL_TRAIN_CACHE = os.path.join(WORKING_DIR, "residual_train.parquet")
    RESIDUAL_VAL_CACHE = os.path.join(WORKING_DIR, "residual_val.parquet")

    # Tokenizer Model Paths
    # We use a custom BPE tokenizer for the target (normalized) text
    TOKENIZER_PREFIX = os.path.join(WORKING_DIR, "bpe_ru_target")

    # Model Checkpoint
    TRANSFORMER_CHECKPOINT = os.path.join(WORKING_DIR, "transformer_residual_best.pth")

    # Output
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # ==========================================
    # 2. Data Processing & Residual Generation
    # ==========================================
    # K-Fold Cross-Validation for generating training residuals
    # High K means more training data for the residual model but slower generation
    K_FOLDS = 5

    # Semiotic Filter
    # Regex to identify tokens that are potential candidates for normalization
    # (Digits or Latin characters). Used to filter the residual dataset.
    SEMIOTIC_REGEX = r"[0-9a-zA-Z]"

    # Context Window
    # Number of characters to include from previous and next tokens for context
    CONTEXT_WINDOW_CHARS = 20

    # Sequence Limits
    MAX_INPUT_LEN = 128  # Max chars for encoder input (Context + Target)
    MAX_OUTPUT_LEN = 64  # Max subwords for decoder output

    # ==========================================
    # 3. Model Architecture (Tier 2 Transformer)
    # ==========================================
    # Character-based Encoder -> Subword-based Decoder

    # Encoder (Input is raw characters)
    CHAR_VOCAB_SIZE = 512  # Sufficient for ASCII + Cyrillic + Symbols

    # Decoder (Output is BPE subwords)
    TARGET_VOCAB_SIZE = 4000  # Compact vocabulary for normalized number words

    # Transformer Dimensions
    D_MODEL = 256
    NHEAD = 4
    NUM_ENCODER_LAYERS = 4
    NUM_DECODER_LAYERS = 4
    DIM_FEEDFORWARD = 1024
    DROPOUT = 0.1

    # ==========================================
    # 4. Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 128
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 15
    EARLY_STOPPING_PATIENCE = 3
    GRADIENT_CLIP_VAL = 1.0

    # Label Smoothing
    LABEL_SMOOTHING = 0.1

    # ==========================================
    # 5. Debugging
    # ==========================================
    DEBUG = False
    DEBUG_SIZE = 50000  # Limit dataset size when debugging

    @classmethod
    def setup(cls):
        """Ensure all necessary working directories exist."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.HFBB_CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize environment
Config.setup()
