import os


class Config:
    """
    Central configuration for the Question Answering task.
    Handles hyperparameters, file paths, and directory setup for
    Task-Adaptive Pretraining (TAPT) and QA Fine-tuning.
    """

    # --------------------------------------------------------------------------
    # General Settings
    # --------------------------------------------------------------------------
    SEED_LIST = [42, 43, 44]  # Seeds for ensemble training
    NUM_WORKERS = 2  # Number of dataloader workers
    DEVICE = "cuda"  # Target device

    # --------------------------------------------------------------------------
    # Model Architecture
    # --------------------------------------------------------------------------
    MODEL_CHECKPOINT = "xlm-roberta-base"

    # --------------------------------------------------------------------------
    # Tokenizer / Sliding Window Strategy
    # --------------------------------------------------------------------------
    MAX_LENGTH = 384
    DOC_STRIDE = 128

    # --------------------------------------------------------------------------
    # Training Hyperparameters (QA Fine-tuning)
    # --------------------------------------------------------------------------
    BATCH_SIZE = 16  # Strictly set to 16 as per lessons learned
    EPOCHS = 10  # 10 epochs for convergence
    LEARNING_RATE = 2e-5  # Standard transformer LR
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    MAX_GRAD_NORM = 1.0

    # --------------------------------------------------------------------------
    # TAPT Hyperparameters (Task-Adaptive Pretraining)
    # --------------------------------------------------------------------------
    TAPT_EPOCHS = 3  # Short pre-training adaptation
    TAPT_BATCH_SIZE = 8  # Slightly smaller batch for MLM memory overhead if needed
    TAPT_LEARNING_RATE = 2e-5
    MLM_PROBABILITY = 0.15

    # --------------------------------------------------------------------------
    # File Paths & Directories
    # --------------------------------------------------------------------------
    # Read-only input directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata file paths
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Working directory for this specific experiment iteration
    WORKING_DIR = "./working/idea_10"

    # QA specific paths
    QA_CACHE_DIR = os.path.join(WORKING_DIR, "qa_cache")
    QA_MODELS_DIR = os.path.join(WORKING_DIR, "qa_models")

    # TAPT specific paths
    TAPT_CACHE_DIR = os.path.join(WORKING_DIR, "tapt_cache")
    TAPT_OUTPUT_DIR = os.path.join(WORKING_DIR, "tapt_model_finetuned")

    # Submission paths
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    @classmethod
    def setup(cls):
        """
        Creates the necessary directory structure for the experiment.
        Should be called at the start of any script using this config.
        """
        # Create working directories
        os.makedirs(cls.WORKING_DIR, exist_ok=True)

        # QA directories
        os.makedirs(cls.QA_CACHE_DIR, exist_ok=True)
        os.makedirs(cls.QA_MODELS_DIR, exist_ok=True)

        # TAPT directories
        os.makedirs(cls.TAPT_CACHE_DIR, exist_ok=True)
        os.makedirs(cls.TAPT_OUTPUT_DIR, exist_ok=True)

        # Submission directory
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
