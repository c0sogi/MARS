import os
import torch
import random
import numpy as np


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Config:
    """
    Centralized configuration for the DeBERTa-v3-small essay scoring model.
    """

    # General Settings
    seed = 42
    debug = False  # Set to True to run on a small subset for debugging

    # Model Settings
    model_name = "microsoft/deberta-v3-small"
    max_length = 1024
    num_labels = 1  # Regression output

    # Training Hyperparameters
    train_batch_size = 4
    valid_batch_size = 8
    epochs = 4
    learning_rate = 2e-5
    weight_decay = 0.01
    max_grad_norm = 10.0

    # Scheduler Settings
    scheduler_type = "linear"
    warmup_ratio = 0.1

    # Hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 4

    # Directories and Paths
    input_dir = "./input"
    metadata_dir = "./metadata"
    working_dir = "./working/idea_2"
    submission_dir = "./submission"

    # Ensure directories exist
    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(submission_dir, exist_ok=True)

    # Data Paths (using metadata splits)
    train_path = os.path.join(metadata_dir, "train.csv")
    val_path = os.path.join(metadata_dir, "val.csv")
    test_path = os.path.join(metadata_dir, "test.csv")

    # Output Paths
    model_save_path = os.path.join(working_dir, "deberta_model.pth")

    # Cache Paths (for processed datasets)
    train_cache_path = os.path.join(working_dir, "train_processed.parquet")
    val_cache_path = os.path.join(working_dir, "val_processed.parquet")
    test_cache_path = os.path.join(working_dir, "test_processed.parquet")

    # Submission Path
    submission_path = os.path.join(submission_dir, "submission.csv")
