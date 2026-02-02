import os
import torch


class Config:
    # =========================================================================
    # General Configuration
    # =========================================================================
    project_name = "tweet-sentiment-extraction"
    idea_name = "idea_6"
    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 4

    # =========================================================================
    # File Paths & Directories
    # =========================================================================
    base_dir = "./"
    input_dir = os.path.join(base_dir, "input")
    metadata_dir = os.path.join(base_dir, "metadata")

    # Working directory for this specific idea (caching, checkpoints)
    working_dir = os.path.join(base_dir, "working", idea_name)

    # Submission directory
    submission_dir = os.path.join(base_dir, "submission")

    # Input Data Paths (using generated metadata)
    train_path = os.path.join(metadata_dir, "train_metadata.csv")
    val_path = os.path.join(metadata_dir, "validation_metadata.csv")
    test_path = os.path.join(metadata_dir, "test_metadata.csv")
    sample_submission_path = os.path.join(input_dir, "sample_submission.csv")

    # Output Artifacts
    model_save_path = os.path.join(working_dir, "best_model.bin")
    submission_path = os.path.join(submission_dir, "submission.csv")

    # =========================================================================
    # Data Processing & Tokenizer
    # =========================================================================
    model_name = "microsoft/deberta-v3-base"
    max_len = 128

    # Preprocessing Flags
    normalize_text = True  # Normalize whitespace before tokenization/extraction
    filter_neutral = True  # Exclude neutral tweets from training

    # Target Generation
    target_smoothing_sigma = 1.0  # Sigma for Gaussian smoothing of start/end targets

    # =========================================================================
    # Model Architecture
    # =========================================================================
    hidden_size = 768  # DeBERTa-base hidden size
    n_pooling_layers = 4  # Number of last hidden layers to aggregate

    # Convolutional Head
    cnn_kernel_size = 3
    cnn_out_channels = 768
    dropout = 0.1

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    debug = False  # If True, runs on a small subset
    debug_sample_size = 500

    epochs = 5
    train_batch_size = 32
    valid_batch_size = 64

    learning_rate = 2e-5
    weight_decay = 0.01
    clip_grad_norm = 1.0

    scheduler = "linear"  # Learning rate scheduler
    warmup_ratio = 0.1
    early_stopping_patience = 2

    # =========================================================================
    # Setup Helper
    # =========================================================================
    @staticmethod
    def setup_dirs():
        """Creates necessary directories for outputs and cache."""
        os.makedirs(Config.working_dir, exist_ok=True)
        os.makedirs(Config.submission_dir, exist_ok=True)


# Initialize directories upon import
Config.setup_dirs()
