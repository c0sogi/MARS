import os
import torch


class Config:
    """
    Configuration class for the Hindi/Tamil Question Answering task.
    Centralizes all hyperparameters, file paths, and model settings.
    """

    # --------------------------------------------------------------------------
    # General Configuration
    # --------------------------------------------------------------------------
    SEED = 42
    SEEDS = [42, 43, 44]  # Seeds for ensembling
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 2  # Number of dataloader workers

    # Debugging flags to control dataset size
    DEBUG = False
    MAX_TRAIN_SAMPLES = None  # Use None for full dataset, or int for debugging
    MAX_VAL_SAMPLES = None  # Use None for full dataset, or int for debugging

    # --------------------------------------------------------------------------
    # Directory Structure & Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for this specific experimental run
    WORKING_DIR = "./working/idea_6"

    # Cache directories for intermediate data (Parquet/PT files)
    TAPT_CACHE_DIR = os.path.join(WORKING_DIR, "tapt_cache")
    QA_CACHE_DIR = os.path.join(WORKING_DIR, "qa_cache")

    # Output directories for model checkpoints
    TAPT_OUTPUT_DIR = os.path.join(WORKING_DIR, "tapt_model_finetuned")
    QA_OUTPUT_DIR = os.path.join(WORKING_DIR, "qa_models")

    # Submission directory
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Input Data Files (using generated metadata)
    TRAIN_FILE = os.path.join(METADATA_DIR, "train.csv")
    VAL_FILE = os.path.join(METADATA_DIR, "val.csv")
    TEST_FILE = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_FILE = os.path.join(INPUT_DIR, "sample_submission.csv")

    # --------------------------------------------------------------------------
    # Model Architecture & Tokenization
    # --------------------------------------------------------------------------
    MODEL_CHECKPOINT = "xlm-roberta-base"

    # Sliding Window Hyperparameters
    MAX_LENGTH = 384
    DOC_STRIDE = 128

    # --------------------------------------------------------------------------
    # Task-Adaptive Pretraining (TAPT) Hyperparameters
    # --------------------------------------------------------------------------
    # Training on domain text (MLM) before QA fine-tuning
    TAPT_EPOCHS = 5
    TAPT_BATCH_SIZE = 8
    TAPT_LEARNING_RATE = 2e-5
    TAPT_WEIGHT_DECAY = 0.01
    TAPT_MLM_PROBABILITY = 0.15

    # --------------------------------------------------------------------------
    # Question Answering (QA) Fine-Tuning Hyperparameters
    # --------------------------------------------------------------------------
    EPOCHS = 10
    TRAIN_BATCH_SIZE = 8  # Small batch size coupled with stratified sampling
    EVAL_BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    MAX_GRAD_NORM = 1.0

    # --------------------------------------------------------------------------
    # Helper Methods
    # --------------------------------------------------------------------------
    @staticmethod
    def create_directories():
        """
        Creates the necessary directory structure for the experiment.
        """
        directories = [
            Config.WORKING_DIR,
            Config.TAPT_CACHE_DIR,
            Config.QA_CACHE_DIR,
            Config.TAPT_OUTPUT_DIR,
            Config.QA_OUTPUT_DIR,
            Config.SUBMISSION_DIR,
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)


# Automatically create directories when config is imported
Config.create_directories()
