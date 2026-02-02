import os
import torch
import random
import numpy as np


class Config:
    """
    Configuration class for the Tweet Sentiment Extraction task.
    Implements parameters for the Two-Stage Self-Distillation pipeline
    using microsoft/deberta-v3-large.
    """

    # ==========================
    # General Settings
    # ==========================
    seed = 42
    n_folds = 5
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    debug = False  # Set to True to run on a small subset for debugging

    # ==========================
    # Data Paths
    # ==========================
    # Using metadata paths to ensure consistent stratified splits
    train_path = "./metadata/train.csv"
    val_path = "./metadata/val.csv"  # Explicit validation set from metadata
    test_path = "./metadata/test.csv"
    sample_submission_path = "./input/sample_submission.csv"

    # Output directory for artifacts (models, cache, predictions)
    output_dir = "./working/idea_13/"

    # ==========================
    # Model Architecture
    # ==========================
    model_name = "microsoft/deberta-v3-large"
    max_len = 128  # Max char len is 141, 128 tokens is sufficient

    # ==========================
    # Training Hyperparameters
    # ==========================
    train_batch_size = 8
    valid_batch_size = 16
    epochs = 4
    lr = 1e-5
    scheduler = "linear"
    warmup_ratio = 0.1
    weight_decay = 0.01
    max_grad_norm = 1.0

    # ==========================
    # Distillation & Loss
    # ==========================
    # Stage 1 (Teacher): CrossEntropy with Label Smoothing
    label_smoothing = 0.1

    # Stage 2 (Student): Distillation Loss
    # Loss = alpha * CE + (1 - alpha) * KL_Div
    distillation_alpha = 0.5
    temperature = 1.0

    @classmethod
    def setup(cls):
        """
        Initializes the environment:
        1. Creates necessary output directories.
        2. Sets random seeds for reproducibility.
        3. Configures environment variables.
        """
        # Create output directory
        os.makedirs(cls.output_dir, exist_ok=True)

        # Set random seeds
        random.seed(cls.seed)
        np.random.seed(cls.seed)
        torch.manual_seed(cls.seed)
        torch.cuda.manual_seed(cls.seed)

        # Ensure deterministic behavior in CuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Disable tokenizer parallelism to prevent deadlocks in DataLoaders
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        print(f"System setup complete. Output directory: {cls.output_dir}")
        print(f"Device: {cls.device}")
