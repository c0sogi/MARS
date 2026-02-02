import os


class Config:
    # --- General Configuration ---
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SAMPLE_SIZE = 100
    NUM_WORKERS = 4

    # --- Paths ---
    # Input data (metadata contains the stratified splits)
    METADATA_DIR = "./metadata"
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Working directory for artifacts
    WORKING_DIR = "./working/idea_4"
    MODEL_OUTPUT_DIR = os.path.join(WORKING_DIR, "models")
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # Submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # --- Semantic Branch (DeBERTa-v3-Large) ---
    MODEL_NAME = "microsoft/deberta-v3-large"
    MAX_LENGTH = 1024

    # Training Hyperparameters
    # A100 40GB allows small batch size for Large model @ 1024 tokens.
    # Using Gradient Accumulation to achieve effective batch size.
    TRAIN_BATCH_SIZE = 2
    VALID_BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 8  # Effective Batch Size = 2 * 8 = 16

    LEARNING_RATE = 1e-5  # Uniform learning rate
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0
    NUM_EPOCHS = 4
    WARMUP_RATIO = 0.1
    LOSS_FN = "SmoothL1Loss"
    USE_FP16 = True  # Mixed precision training

    # --- Lexical Branch (TF-IDF + Ridge) ---
    TFIDF_PARAMS = {
        "ngram_range": (1, 3),
        "min_df": 3,
        "sublinear_tf": True,
        "use_idf": True,
        "strip_accents": "unicode",
        "analyzer": "word",
        "token_pattern": r"\w{1,}",
    }

    # --- Stacking / Meta-Learner ---
    N_FOLDS = 5

    # --- Validation ---
    # Number of steps to wait before early stopping
    PATIENCE = 3
