import os


class Config:
    """
    Global configuration for the Hindi/Tamil Question Answering Task.
    Includes paths, model parameters, and training hyperparameters for both
    Task-Adaptive Pretraining (TAPT) and QA Fine-tuning.
    """

    # =========================================================================
    # Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_4"
    SUBMISSION_DIR = "./submission"

    # =========================================================================
    # Data Paths (Metadata)
    # =========================================================================
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # =========================================================================
    # Model Architecture
    # =========================================================================
    MODEL_CHECKPOINT = "xlm-roberta-base"

    # =========================================================================
    # Data Processing / Tokenizer
    # =========================================================================
    MAX_LENGTH = 384
    DOC_STRIDE = 128

    # =========================================================================
    # Training Hyperparameters (QA Fine-Tuning)
    # =========================================================================
    BATCH_SIZE = 16
    EPOCHS = 10
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    SEEDS = [42, 43, 44]  # Seeds for ensemble members

    # =========================================================================
    # TAPT Hyperparameters (Task-Adaptive Pretraining)
    # =========================================================================
    # Using a small batch size to ensure frequent updates as per strategy
    TAPT_BATCH_SIZE = 8
    TAPT_EPOCHS = 3
    MLM_PROBABILITY = 0.15
    TAPT_LEARNING_RATE = 2e-5

    # =========================================================================
    # Artifact Storage Paths
    # =========================================================================
    # Cache directories for processed datasets (parquet/pt files)
    TAPT_CACHE_DIR = os.path.join(WORKING_DIR, "tapt_cache")
    QA_CACHE_DIR = os.path.join(WORKING_DIR, "qa_cache")

    # Model checkpoint directories
    TAPT_MODEL_DIR = os.path.join(WORKING_DIR, "tapt_model_finetuned")
    QA_MODELS_DIR = os.path.join(WORKING_DIR, "qa_models")

    @classmethod
    def setup(cls):
        """
        Creates the necessary directory structure for the experiment.
        Should be called at the start of the pipeline.
        """
        dirs = [
            cls.WORKING_DIR,
            cls.SUBMISSION_DIR,
            cls.TAPT_CACHE_DIR,
            cls.QA_CACHE_DIR,
            cls.TAPT_MODEL_DIR,
            cls.QA_MODELS_DIR,
        ]

        for d in dirs:
            os.makedirs(d, exist_ok=True)
