import os


class Config:
    """
    Configuration class for the Hindi/Tamil Question Answering task.
    Centralizes hyperparameters, model settings, and file paths.
    """

    # =========================================================================
    # Model Settings
    # =========================================================================
    # Using MuRIL (Multilingual Representations for Indian Languages)
    # as the backbone for better alignment with Hindi and Tamil text.
    model_checkpoint = "google/muril-base-cased"

    # =========================================================================
    # Data Processing / Tokenization
    # =========================================================================
    # Windowing parameters to handle long contexts
    max_length = 384
    doc_stride = 128

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    # Optimized for the available compute and dataset size
    batch_size = 8
    epochs = 5
    learning_rate = 3e-5
    weight_decay = 0.01
    warmup_ratio = 0.1
    seed = 42

    # Ensemble Strategy
    n_folds = 5

    # =========================================================================
    # File Paths
    # =========================================================================
    # Input data paths (using metadata to ensure correct splits)
    train_data_path = "./metadata/train.csv"
    val_data_path = "./metadata/val.csv"
    test_data_path = "./metadata/test.csv"
    sample_submission_path = "./input/sample_submission.csv"

    # Working directory for caching intermediate files (features, models)
    working_dir = "./working/idea_3"

    # Output directory for the final submission file
    submission_dir = "./submission"
    submission_file = os.path.join(submission_dir, "submission.csv")

    # Cache file paths for processed datasets (Parquet format)
    train_cache_path = os.path.join(working_dir, "train_processed.parquet")
    val_cache_path = os.path.join(working_dir, "val_processed.parquet")
    test_cache_path = os.path.join(working_dir, "test_processed.parquet")

    def __init__(self):
        """
        Initialize configuration and ensure necessary directories exist.
        """
        # Create working directory for caching
        os.makedirs(self.working_dir, exist_ok=True)

        # Create submission directory
        os.makedirs(self.submission_dir, exist_ok=True)
