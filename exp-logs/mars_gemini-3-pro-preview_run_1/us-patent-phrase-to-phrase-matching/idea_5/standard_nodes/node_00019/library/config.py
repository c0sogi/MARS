import os
import torch
import random
import numpy as np


class Config:
    # =========================================================================
    # General Settings
    # =========================================================================
    seed = 42
    debug = False  # Set to True for fast debugging
    debug_sample_size = 100  # Number of samples to use in debug mode
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================================================================
    # Paths
    # =========================================================================
    input_dir = "./input"
    metadata_dir = "./metadata"
    working_dir = "./working/idea_5"

    # Data Paths
    train_path = os.path.join(metadata_dir, "train.csv")
    val_path = os.path.join(metadata_dir, "val.csv")
    test_path = os.path.join(metadata_dir, "test.csv")
    cpc_context_path = os.path.join(input_dir, "description.md")

    # Output Paths
    models_dir = os.path.join(working_dir, "models")
    predictions_dir = os.path.join(working_dir, "predictions")
    submission_dir = "./submission"
    submission_path = os.path.join(submission_dir, "submission.csv")

    # =========================================================================
    # Model Architecture
    # =========================================================================
    model_name = "microsoft/deberta-v3-large"
    max_len = 128  # Context + Anchor + Target
    target_col = "score"

    # Multi-Sample Dropout Settings
    num_msd_rounds = 5
    fc_dropout = 0.2

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    num_folds = 5
    epochs = 4
    train_batch_size = 8  # Adjusted for A100 memory with large model
    valid_batch_size = 16
    gradient_accumulation_steps = 1
    max_grad_norm = 10.0

    # Optimization (LLRD & Scheduler)
    lr = 2e-5  # Base learning rate
    min_lr = 1e-6
    weight_decay = 0.01
    layer_decay = 0.9  # Layer-wise learning rate decay factor
    warmup_ratio = 0.1
    scheduler = "cosine"  # 'cosine' or 'linear'

    # Loss
    loss_fn = "MSE"  # Mean Squared Error for Pearson correlation task

    # =========================================================================
    # Utils
    # =========================================================================
    @classmethod
    def create_dirs(cls):
        """Creates necessary directories for outputs."""
        os.makedirs(cls.working_dir, exist_ok=True)
        os.makedirs(cls.models_dir, exist_ok=True)
        os.makedirs(cls.predictions_dir, exist_ok=True)
        os.makedirs(cls.submission_dir, exist_ok=True)

    @classmethod
    def setup_system(cls):
        """Sets fixed seeds for reproducibility."""
        random.seed(cls.seed)
        np.random.seed(cls.seed)
        torch.manual_seed(cls.seed)
        torch.cuda.manual_seed_all(cls.seed)

        # Ensure deterministic behavior where possible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Create directories
        cls.create_dirs()
