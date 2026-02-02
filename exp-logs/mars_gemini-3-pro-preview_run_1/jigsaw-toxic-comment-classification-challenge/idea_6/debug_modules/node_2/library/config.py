import os
import torch


class Config:
    """
    Configuration class for the Toxicity Classification task.
    Centralizes all hyperparameters, paths, and system settings.
    """

    # ==========================================
    # General Settings
    # ==========================================
    seed = 42
    debug = False  # Set to True to run on a small subset for testing
    debug_subset_size = 1000  # Number of samples to use in debug mode

    # ==========================================
    # Data & Paths
    # ==========================================
    input_dir = "./input"
    metadata_dir = "./metadata"
    working_dir = "./working/idea_6"
    submission_dir = "./submission"
    submission_path = os.path.join(submission_dir, "submission.csv")

    # Metadata file paths
    train_meta_path = os.path.join(metadata_dir, "train.csv")
    val_meta_path = os.path.join(metadata_dir, "val.csv")
    test_meta_path = os.path.join(metadata_dir, "test.csv")

    # Raw data paths (for merging text content)
    train_data_path = os.path.join(input_dir, "train.csv")
    test_data_path = os.path.join(input_dir, "test.csv")

    # Target Columns
    target_cols = [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ]
    num_classes = len(target_cols)

    # ==========================================
    # Model Architecture
    # ==========================================
    model_name = "microsoft/deberta-v3-base"
    max_len = 300
    hidden_size = 768
    dropout = 0.1

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    epochs = 4
    train_batch_size = 32  # Conservative batch size for A100
    valid_batch_size = 64

    # Optimizer & Scheduler
    learning_rate = 2e-5
    weight_decay = 0.01
    max_grad_norm = 1.0  # Gradient clipping
    scheduler_type = "OneCycleLR"
    pct_start = 0.1  # Warmup percentage for OneCycleLR

    # ==========================================
    # System & Hardware
    # ==========================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 4  # Number of dataloader workers
    pin_memory = True

    @classmethod
    def create_dirs(cls):
        """
        Creates necessary directories for working and submission.
        """
        os.makedirs(cls.working_dir, exist_ok=True)
        os.makedirs(cls.submission_dir, exist_ok=True)

    @classmethod
    def print_config(cls):
        """
        Prints the current configuration.
        """
        print("=" * 30)
        print("Current Configuration")
        print("=" * 30)
        for attr in dir(cls):
            if not attr.startswith("__") and not callable(getattr(cls, attr)):
                val = getattr(cls, attr)
                print(f"{attr}: {val}")
        print("=" * 30)
