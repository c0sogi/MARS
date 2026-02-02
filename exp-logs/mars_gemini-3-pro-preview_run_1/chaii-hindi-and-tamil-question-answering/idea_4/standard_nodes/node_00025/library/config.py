import os
import torch


class Config:
    """
    Configuration class for the Hindi/Tamil Question Answering Task.
    Centralizes hyperparameters, file paths, and model settings.
    """

    # --------------------------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------------------------
    SEED = 42

    # --------------------------------------------------------------------------
    # File Paths & Directories
    # --------------------------------------------------------------------------
    # Input data is read from the generated metadata to ensure consistent splits
    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Working directory for caching processed data and model checkpoints
    IDEA_NAME = "idea_5"
    WORKING_DIR = os.path.join("./working", IDEA_NAME)

    # Submission output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Model Architecture
    # --------------------------------------------------------------------------
    # Using XLM-Roberta-Base (Cite Lesson 00024)
    MODEL_CHECKPOINT = "xlm-roberta-base"

    # --------------------------------------------------------------------------
    # Data Processing & Tokenization
    # --------------------------------------------------------------------------
    MAX_LENGTH = 384
    DOC_STRIDE = 128

    # Negative Sampling Strategy for Training
    # Ratio of negative windows (no answer) to positive windows (contains answer).
    # 2.0 means we keep 2 negative samples for every 1 positive sample.
    # This helps mitigate the severe class imbalance in sliding window QA.
    NEGATIVE_SAMPLING_RATIO = 2.0

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    N_FOLDS = 5
    BATCH_SIZE = 8
    EPOCHS = 7
    LEARNING_RATE = 3e-5
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    MAX_GRAD_NORM = 1.0

    # Early Stopping
    PATIENCE = 2

    # --------------------------------------------------------------------------
    # Hardware & Compute
    # --------------------------------------------------------------------------
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 2

    # --------------------------------------------------------------------------
    # Debugging
    # --------------------------------------------------------------------------
    # If True, runs the pipeline on a small subset of data
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 50

    @classmethod
    def setup(cls):
        """
        Creates the necessary working and submission directories.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        print(f"Directories ensured: {cls.WORKING_DIR}, {cls.SUBMISSION_DIR}")
