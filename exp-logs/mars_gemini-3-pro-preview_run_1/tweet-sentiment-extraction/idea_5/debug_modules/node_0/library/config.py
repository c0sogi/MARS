import os
import torch
import random
import numpy as np


class Config:
    """
    Configuration class for Sentiment Extraction task.
    Encapsulates all hyperparameters for Model, Data, and Training.
    """

    # ==========================
    # General Settings
    # ==========================
    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 4

    # ==========================
    # Paths & Directories
    # ==========================
    # Input Metadata (Pre-generated)
    meta_dir = "./metadata"
    train_path = os.path.join(meta_dir, "train_metadata.csv")
    val_path = os.path.join(meta_dir, "validation_metadata.csv")
    test_path = os.path.join(meta_dir, "test_metadata.csv")

    # Output & Working Directories
    base_dir = "./working/idea_5"
    model_save_path = os.path.join(base_dir, "best_model.bin")
    output_submission_path = "./submission/submission.csv"

    # Ensure directories exist
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_submission_path), exist_ok=True)

    # ==========================
    # Model Architecture
    # ==========================
    model_name = "microsoft/deberta-v3-base"
    max_len = 128  # Covers max char length of 141 comfortably
    dropout = 0.1

    # Custom Head Parameters
    n_pool_layers = 4  # Weighted Layer Pooling using last 4 layers
    cnn_kernel_size = 3  # 1D Conv kernel size for span boundary detection

    # ==========================
    # Training Hyperparameters
    # ==========================
    epochs = 8  # Extended training time for LLRD adaptation
    train_batch_size = 16
    valid_batch_size = 32

    # Optimization
    learning_rate = 2e-5
    min_lr = 1e-6
    weight_decay = 0.01
    max_grad_norm = 1.0

    # Scheduler
    scheduler_type = "cosine"
    warmup_ratio = 0.1

    # ==========================
    # Advanced Strategies
    # ==========================
    # Layer-wise Learning Rate Decay
    llrd_decay = 0.9

    # Adversarial Weight Perturbation (AWP)
    use_awp = True
    awp_start_epoch = 2  # Start AWP after the first epoch (1-based count logic)
    awp_eps = 1e-2
    awp_lr = 1e-4

    # Loss & Targets
    gaussian_sigma = 1.0  # Sigma for Gaussian smoothing of start/end targets
    aux_loss_weight = 0.5  # Weight for the auxiliary dense binary classification head

    # Data Processing Logic
    train_on_neutral = False  # Strategy: Exclude neutral tweets from training

    # Logging
    print_freq = 50


def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # Ensure hash consistency
    os.environ["PYTHONHASHSEED"] = str(seed)


# Initialize seed on import
set_seed(Config.seed)
