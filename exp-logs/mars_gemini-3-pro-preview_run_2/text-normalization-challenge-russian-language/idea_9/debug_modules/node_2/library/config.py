import os
import torch


class Config:
    """
    Configuration for the Text Normalization Task (Idea 9: Confidence-Gated Curriculum Cascade).
    Defines paths, hyperparameters, and constants for the pipeline.
    """

    # ==========================================
    # 1. PATHS & DIRECTORIES
    # ==========================================
    # Root directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_9"
    SUBMISSION_DIR = "./submission"  # Root submission directory

    # Input Data Files (from metadata generation)
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.csv")
    # Note: Using the file listed in Dataset Information
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "ru_sample_submission_2.csv")

    # Output Directories (Cache & Artifacts)
    CACHE_DIR = os.path.join(WORKING_DIR, "data_cache")
    HFBB_CACHE_DIR = os.path.join(WORKING_DIR, "hfbb_cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    TOKENIZER_DIR = os.path.join(WORKING_DIR, "tokenizers")

    # Specific Artifact Paths
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "transformer_best.pth")
    TARGET_TOKENIZER_MODEL = os.path.join(TOKENIZER_DIR, "bpe_ru_target.model")
    TARGET_TOKENIZER_VOCAB = os.path.join(TOKENIZER_DIR, "bpe_ru_target.vocab")
    CHAR_VOCAB_PATH = os.path.join(TOKENIZER_DIR, "char_vocab.json")
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # 2. DATA & PREPROCESSING
    # ==========================================
    SEED = 42

    # Context Window for Transformer: [Prev2, Prev1, Target, Next1, Next2]
    CONTEXT_WINDOW = 2

    # Sequence Lengths
    # Character encoder needs enough space for center token + context
    MAX_LEN_CHAR = 128
    # Subword decoder needs enough space for normalized output
    MAX_LEN_SUBWORD = 128

    # Tokenizer Settings
    TARGET_VOCAB_SIZE = (
        4000  # Compact vocabulary for normalized text (mostly words/numbers)
    )

    # Curriculum / Dataset Construction
    # Percentage of high-confidence simple tokens (anchors) to mix with residuals
    ANCHOR_RATIO = 0.20
    # Threshold for Unigram confidence to consider a correct match "ambiguous" (Hard Positive)
    AMBIGUITY_THRESHOLD = 0.90
    # Whether to upsample rare classes (Money, Measure, etc.) in the training set
    UPSAMPLE_RARE_CLASSES = True

    # ==========================================
    # 3. MODEL ARCHITECTURE (TIER 2 TRANSFORMER)
    # ==========================================
    # Character-to-Subword Transformer
    D_MODEL = 512
    NHEAD = 8
    NUM_ENCODER_LAYERS = 6
    NUM_DECODER_LAYERS = 6
    DIM_FEEDFORWARD = 2048
    DROPOUT = 0.1

    # ==========================================
    # 4. TRAINING HYPERPARAMETERS
    # ==========================================
    BATCH_SIZE = 256  # A100 40GB allows large batches
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.01
    EPOCHS = 20  # Max epochs (controlled by early stopping)
    WARMUP_STEPS = 2000
    LABEL_SMOOTHING = 0.1
    PATIENCE = 3  # Early stopping patience
    GRAD_CLIP = 1.0

    # ==========================================
    # 5. CASCADE / INFERENCE LOGIC
    # ==========================================
    # Tier 1 (HFBB) Settings
    # If Unigram confidence > this, we trust the statistical model (unless semiotic checks fail)
    CONFIDENCE_THRESHOLD = 0.95

    # ==========================================
    # 6. HARDWARE & EXECUTION
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 12  # 12 vCPUs available
    PIN_MEMORY = True

    # Debugging / Development
    DEBUG = False  # Set True to use a small subset of data
    DEBUG_SIZE = 50000  # Size of subset if DEBUG is True

    @classmethod
    def setup(cls):
        """
        Ensure all necessary working directories exist.
        """
        directories = [
            cls.WORKING_DIR,
            cls.CACHE_DIR,
            cls.HFBB_CACHE_DIR,
            cls.CHECKPOINT_DIR,
            cls.TOKENIZER_DIR,
            cls.SUBMISSION_DIR,
        ]
        for d in directories:
            os.makedirs(d, exist_ok=True)


# Initialize directories immediately upon import
Config.setup()
