import os
import torch


class Config:
    """
    Configuration class for the Locate-and-Fill model pipeline.
    Centralizes all hyperparameters, paths, and system settings.
    """

    # ---------------------------------------------------------
    # General Settings
    # ---------------------------------------------------------
    SEED = 42
    PROJECT_NAME = "idea_1_locate_and_fill"

    # Debugging / Development
    # Set DEBUG to True to train on a small subset of data for quick validation
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 500_000

    # Cite solution_lesson_node_00005: Train on 1M unique samples for 1 epoch
    TRAIN_SAMPLE_SIZE = 1_000_000

    # ---------------------------------------------------------
    # File Paths
    # ---------------------------------------------------------
    # Input Metadata (Read-Only)
    METADATA_DIR = "./metadata"
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Working Directory (Read/Write)
    WORKING_DIR = os.path.join("./working", "idea_1")
    OUTPUT_DIR = os.path.join(WORKING_DIR, "outputs")
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # Model Checkpoints
    LOCATOR_MODEL_DIR = os.path.join(OUTPUT_DIR, "locator_checkpoints")
    FILLER_MODEL_DIR = os.path.join(OUTPUT_DIR, "filler_checkpoints")

    # Submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ---------------------------------------------------------
    # Model Architecture
    # ---------------------------------------------------------
    # Using DistilBERT for efficiency within the 24h time limit
    MODEL_BACKBONE = "distilbert-base-uncased"
    TOKENIZER_NAME = "distilbert-base-uncased"

    # Max sequence length (EDA showed mean length ~25 words, max < 128 covers vast majority)
    MAX_LEN = 128

    # ---------------------------------------------------------
    # Training Hyperparameters
    # ---------------------------------------------------------
    # General Training
    NUM_WORKERS = 4  # 12 vCPUs available
    PIN_MEMORY = True

    # Locator Model (Token Classification / Pointer Network)
    LOCATOR_PARAMS = {
        "lr": 2e-5,
        "batch_size": 128,  # A100 40GB can handle large batches
        "epochs": 1,  # Cite solution_lesson_node_00005: Single epoch on larger data
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "early_stopping_patience": 2,
        "grad_clip": 1.0,
        "save_best_only": True,
    }

    # Filler Model (Masked Language Modeling)
    FILLER_PARAMS = {
        "lr": 5e-5,
        "batch_size": 64,
        "epochs": 1,  # Cite solution_lesson_node_00005: Single epoch on larger data
        "weight_decay": 0.01,
        "warmup_ratio": 0.06,
        "mlm_probability": 0.15,  # Standard BERT masking rate
        "early_stopping_patience": 2,
        "save_best_only": True,
    }

    # ---------------------------------------------------------
    # Hardware
    # ---------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """
        Creates necessary directories for outputs and cache.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.LOCATOR_MODEL_DIR, exist_ok=True)
        os.makedirs(cls.FILLER_MODEL_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        print(f"Configuration Setup Complete.")
        print(f"Device: {cls.DEVICE}")
        print(f"Working Directory: {cls.WORKING_DIR}")
