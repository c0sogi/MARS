import os
import torch


class Config:
    """
    Configuration class for the Hindi/Tamil Question Answering Task.
    Centralizes all hyperparameters, file paths, and model settings.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2  # Conservative number of workers for data loading

    # =========================================================================
    # File Paths
    # =========================================================================
    # Input Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Data Files (Using the pre-split metadata)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Directories (All artifacts go to ./working/idea_7)
    WORKING_DIR = "./working/idea_7"

    # Sub-directories for organization
    CACHE_DIR = os.path.join(WORKING_DIR, "qa_cache")
    MODEL_OUTPUT_DIR = os.path.join(WORKING_DIR, "qa_models")
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")

    # TAPT (Task-Adaptive Pretraining) specific paths
    TAPT_CACHE_DIR = os.path.join(WORKING_DIR, "tapt_cache")
    TAPT_OUTPUT_DIR = os.path.join(WORKING_DIR, "tapt_model_finetuned")

    # Final Submission File
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Create directories if they don't exist
    for d in [
        WORKING_DIR,
        CACHE_DIR,
        MODEL_OUTPUT_DIR,
        SUBMISSION_DIR,
        TAPT_CACHE_DIR,
        TAPT_OUTPUT_DIR,
    ]:
        os.makedirs(d, exist_ok=True)

    # =========================================================================
    # Model Architecture & Tokenization
    # =========================================================================
    # Base model
    MODEL_CHECKPOINT = "xlm-roberta-base"

    # Token Classification Head
    # 3 Labels: 0=O (Outside), 1=B-ANS (Begin Answer), 2=I-ANS (Inside Answer)
    NUM_LABELS = 3

    # Sliding Window Parameters
    MAX_LENGTH = 384
    DOC_STRIDE = 128

    # =========================================================================
    # Training Hyperparameters (QA Fine-tuning)
    # =========================================================================
    BATCH_SIZE = 16  # As per "Idea" section requirements
    LEARNING_RATE = 2e-5  # Standard for XLM-R
    WEIGHT_DECAY = 0.01
    EPOCHS = 10  # As per "Idea" section requirements
    WARMUP_RATIO = 0.1
    MAX_GRAD_NORM = 1.0

    # Early Stopping
    PATIENCE = 3

    # Ensemble Strategy
    SEEDS = [42, 43, 44]  # Train 3 models with different seeds

    # =========================================================================
    # TAPT Hyperparameters (Masked Language Modeling)
    # =========================================================================
    TAPT_EPOCHS = 3
    TAPT_BATCH_SIZE = 8  # Smaller batch size for MLM usually
    TAPT_LEARNING_RATE = 2e-5
    MLM_PROBABILITY = 0.15

    # =========================================================================
    # Caching Control
    # =========================================================================
    # Set to True to try loading features from parquet files in CACHE_DIR
    LOAD_CACHED_DATA = True
