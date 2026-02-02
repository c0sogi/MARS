import os
import torch


class Config:
    # ==== File Paths ====
    # Metadata directories (Input)
    METADATA_DIR = "./metadata"
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = "./input/sample_submission.csv"

    # Output directories (Working)
    # We use idea_6 as the specific directory for this iteration
    OUTPUT_DIR = "./working/idea_6"
    MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache Paths
    CACHE_DIR = OUTPUT_DIR
    TRAIN_CACHE_FILE = os.path.join(CACHE_DIR, "train_data.parquet")
    VAL_CACHE_FILE = os.path.join(CACHE_DIR, "val_data.parquet")
    TEST_CACHE_FILE = os.path.join(CACHE_DIR, "test_data.parquet")

    # ==== Model Architecture ====
    MODEL_NAME = "microsoft/deberta-v3-small"
    MAX_LENGTH = 512
    NUM_CLASSES = 3  # Winner A, Winner B, Tie
    USE_SCALAR_FEATURES = True

    # Dropout settings
    HIDDEN_DROPOUT_PROB = 0.1
    ATTENTION_PROBS_DROPOUT_PROB = 0.1

    # ==== Training Hyperparameters ====
    SEED = 42
    EPOCHS = 3
    TRAIN_BATCH_SIZE = 8  # Increased for Small model
    VALID_BATCH_SIZE = 8
    GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = 4 * 4 = 16
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    MAX_GRAD_NORM = 1.0
    EARLY_STOPPING_PATIENCE = 2

    # Optimization
    USE_FP16 = True  # Mixed precision
    USE_GRADIENT_CHECKPOINTING = True  # Save memory for larger batch/model

    # ==== Hardware ====
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    @classmethod
    def setup(cls):
        """Creates necessary output directories."""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        # Cache dir is same as output dir
        os.makedirs(cls.CACHE_DIR, exist_ok=True)


# Run setup immediately when module is imported to ensure directories exist
Config.setup()
