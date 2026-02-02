import os


class Config:
    # General Settings
    SEED = 42
    NUM_WORKERS = 4

    # Model Architecture
    MODEL_NAME = "distilroberta-base"
    HIDDEN_SIZE = 768
    NUM_TARGETS = 30

    # Data Processing
    MAX_LEN = 512  # Maximum sequence length for tokenization

    # Training Hyperparameters
    TRAIN_BATCH_SIZE = 16
    VALID_BATCH_SIZE = 32
    EPOCHS = 5

    # Optimization
    LEARNING_RATE = 2e-5  # Base learning rate for the backbone
    HEAD_LR = 1e-3  # Higher learning rate for the regression head
    LLRD_DECAY = 0.95  # Layer-wise learning rate decay factor
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    MAX_GRAD_NORM = 1.0

    # Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_15"
    SUBMISSION_DIR = "./submission"

    # Input File Paths (using generated metadata)
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output File Paths
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache File Paths (Parquet format preferred over pickle)
    TRAIN_CACHE_PATH = os.path.join(WORKING_DIR, "train_processed.parquet")
    VAL_CACHE_PATH = os.path.join(WORKING_DIR, "val_processed.parquet")
    TEST_CACHE_PATH = os.path.join(WORKING_DIR, "test_processed.parquet")

    # Ensure necessary directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
