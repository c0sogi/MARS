import os
import torch
import hashlib
import json


class Config:
    # ==========================================
    # Reproducibility
    # ==========================================
    SEED = 42

    # ==========================================
    # Hardware
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Use most available vCPUs for data loading
    NUM_WORKERS = 12
    # A100 40GB allows for larger batch sizes
    BATCH_SIZE = 256

    # ==========================================
    # Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Base working directory
    BASE_WORK_DIR = "./working/idea_14"

    # Input Files (using metadata splits)
    TRAIN_DATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_DATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_DATA = os.path.join(METADATA_DIR, "test.csv")

    # Submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # HFBB (Tier 1) Configuration
    # ==========================================
    # Confidence threshold for Unigram gating
    HFBB_CONFIDENCE_THRESHOLD = 0.99

    # ==========================================
    # Data Processing & Tokenization
    # ==========================================
    # Context window: +/- 2 words around the target
    CONTEXT_WINDOW = 2

    # Max lengths
    # Encoder: Character-level input + context. 256 chars is usually sufficient for 5 tokens.
    MAX_ENC_LEN = 256
    # Decoder: Subword-level output. 128 subwords is plenty for normalized text.
    MAX_DEC_LEN = 128

    # Tokenizer Settings
    # Target BPE vocabulary size (Russian morphology)
    BPE_VOCAB_SIZE = 8000

    # ==========================================
    # Model Architecture (Tier 2: Transformer)
    # ==========================================
    D_MODEL = 512
    NHEAD = 8
    NUM_ENCODER_LAYERS = 6
    NUM_DECODER_LAYERS = 6
    DIM_FEEDFORWARD = 2048
    DROPOUT = 0.1

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.01
    NUM_EPOCHS = 20
    # Early stopping patience
    PATIENCE = 3
    # Warmup steps for scheduler
    WARMUP_STEPS = 2000
    # Label smoothing for regularization
    LABEL_SMOOTHING = 0.1
    # Gradient clipping
    MAX_GRAD_NORM = 1.0

    # ==========================================
    # Soft-Residual Learning Weights
    # ==========================================
    # Weight for "Easy" tokens (Anchors) handled well by Tier 1
    WEIGHT_ANCHOR = 0.1
    # Weight for "Hard" tokens (Residuals) requiring neural adaptation
    WEIGHT_RESIDUAL = 1.0

    # ==========================================
    # Debugging
    # ==========================================
    # Set to True to run on a small subset for testing pipeline
    DEBUG = False
    DEBUG_SIZE = 50000

    # ==========================================
    # Dynamic Configuration Hashing & Paths
    # ==========================================
    @classmethod
    def get_hash(cls):
        """
        Computes a hash of the current configuration to ensure cache invalidation.
        Excludes path-related attributes that don't affect model logic.
        """
        config_dict = {}
        for k, v in cls.__dict__.items():
            # Filter for configuration constants (uppercase)
            if k.isupper() and not k.startswith("__"):
                # Exclude paths and hardware specific settings from hash
                # (Changing the path shouldn't invalidate the data content if params are same)
                if (
                    "DIR" not in k
                    and "PATH" not in k
                    and "WORKERS" not in k
                    and "DEVICE" not in k
                ):
                    config_dict[k] = v

        # Serialize and hash
        config_str = json.dumps(config_dict, sort_keys=True, default=str)
        return hashlib.md5(config_str.encode("utf-8")).hexdigest()[:10]

    @classmethod
    def setup(cls):
        """
        Sets up the artifact directories based on the config hash.
        """
        config_hash = cls.get_hash()

        # Artifact directory specific to this configuration
        cls.ARTIFACT_DIR = os.path.join(cls.BASE_WORK_DIR, config_hash)

        # Sub-directories
        cls.CACHE_DIR = os.path.join(cls.ARTIFACT_DIR, "cache")
        cls.CHECKPOINT_DIR = os.path.join(cls.ARTIFACT_DIR, "checkpoints")
        cls.TOKENIZER_DIR = os.path.join(cls.ARTIFACT_DIR, "tokenizers")

        # Specific file paths
        cls.BPE_MODEL_PREFIX = os.path.join(cls.TOKENIZER_DIR, "bpe_ru_target")
        cls.BEST_MODEL_PATH = os.path.join(cls.CHECKPOINT_DIR, "transformer_best.pth")

        # Create directories
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.TOKENIZER_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize the dynamic paths
Config.setup()
