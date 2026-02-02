import os
import torch


class Config:
    """
    Configuration class for the DeBERTa-v3-base toxicity classification task.
    Centralizes hyperparameters, file paths, and execution settings.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    seed = 42
    debug = False  # Set to True to use a small subset of data for debugging
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================================================================
    # Directory & File Paths
    # =========================================================================
    input_dir = "./input"
    metadata_dir = "./metadata"
    working_dir = "./working/idea_4"
    submission_dir = "./submission"

    # Ensure output directories exist
    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(submission_dir, exist_ok=True)

    # Metadata Files (Defining Train/Val/Test splits)
    train_metadata_path = os.path.join(metadata_dir, "train.csv")
    val_metadata_path = os.path.join(metadata_dir, "val.csv")
    test_metadata_path = os.path.join(metadata_dir, "test.csv")

    # Raw Data Files (Containing text content)
    train_raw_path = os.path.join(input_dir, "train.csv")
    test_raw_path = os.path.join(input_dir, "test.csv")
    sample_submission_path = os.path.join(input_dir, "sample_submission.csv")

    # Cache Files (For processed datasets)
    # Using parquet as requested for caching
    train_cache_path = os.path.join(working_dir, "train_cache.parquet")
    val_cache_path = os.path.join(working_dir, "val_cache.parquet")
    test_cache_path = os.path.join(working_dir, "test_cache.parquet")

    # Output Paths
    model_save_path = os.path.join(working_dir, "model.pth")
    submission_path = os.path.join(submission_dir, "submission.csv")

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================
    model_name = "microsoft/deberta-v3-base"
    max_len = 300
    hidden_size = 768

    # Multi-Sample Dropout Settings
    dropout_samples = 5
    dropout_rate = 0.5

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    epochs = 4
    train_batch_size = 16
    valid_batch_size = 32
    learning_rate = 2e-5
    weight_decay = 0.01
    max_grad_norm = 1.0

    # Scheduler (OneCycleLR)
    scheduler_type = "OneCycleLR"
    pct_start = 0.1
    div_factor = 25.0
    final_div_factor = 1000.0

    # =========================================================================
    # Target Labels
    # =========================================================================
    target_cols = [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ]
    num_classes = len(target_cols)
