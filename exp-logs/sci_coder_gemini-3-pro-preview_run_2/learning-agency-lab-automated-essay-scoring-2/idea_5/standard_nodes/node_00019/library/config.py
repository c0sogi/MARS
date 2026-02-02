import os
import torch


class Config:
    # =========================================================================
    # General Settings
    # =========================================================================
    PROJECT_NAME = "EssayScoring_Idea5"
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset of data
    DEBUG_SAMPLE_SIZE = 100

    # =========================================================================
    # Data Paths
    # =========================================================================
    # Input Metadata
    TRAIN_DATA_PATH = "./metadata/train.csv"
    VAL_DATA_PATH = "./metadata/val.csv"
    TEST_DATA_PATH = "./metadata/test.csv"

    # Output Directories
    WORKING_DIR = "./working/idea_5"
    OUTPUT_DIR = os.path.join(WORKING_DIR, "models")
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Semantic Branch (DeBERTa-v3-Large)
    # =========================================================================
    MODEL_NAME = "microsoft/deberta-v3-large"
    MAX_LENGTH = 1024
    NUM_LABELS = 1

    # Pooling: 'mean', 'max', 'mean_max', 'cls'
    POOLING = "mean_max"

    # Dropout (Set to 0 for regression stability)
    HIDDEN_DROPOUT = 0.0
    ATTENTION_DROPOUT = 0.0

    # =========================================================================
    # Lexical Branch (TF-IDF + Ridge)
    # =========================================================================
    TFIDF_NGRAM_RANGE = (1, 3)
    TFIDF_MIN_DF = 3
    RIDGE_ALPHA = 1.0  # Default, usually tuned via RidgeCV

    # =========================================================================
    # Training Configuration
    # =========================================================================
    N_FOLDS = 5
    EPOCHS = 4

    # Batch Size & Gradient Accumulation
    # Adjusted for A100 40GB with 1024 sequence length
    TRAIN_BATCH_SIZE = 4
    VALID_BATCH_SIZE = 8
    GRAD_ACCUM_STEPS = 4

    # Optimizer & Scheduler
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 10.0
    WARMUP_RATIO = 0.05
    SCHEDULER_TYPE = "cosine"

    # Loss Function
    LOSS_FN = "SmoothL1Loss"
    SMOOTH_L1_BETA = 1.0

    # =========================================================================
    # Adversarial Weight Perturbation (AWP)
    # =========================================================================
    USE_AWP = True
    AWP_START_EPOCH = 1  # 0-indexed: Start from the 2nd epoch (index 1)
    AWP_ADV_LR = 1e-4
    AWP_ADV_EPS = 1e-2
    AWP_ADV_STEP = 1

    # =========================================================================
    # Post-Processing
    # =========================================================================
    USE_NELDER_MEAD = True

    # =========================================================================
    # Hardware & Environment
    # =========================================================================
    NUM_WORKERS = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def create_dirs(cls):
        """Creates necessary directories for outputs, cache, and submission."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Suppress tokenizer parallelism warnings
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
