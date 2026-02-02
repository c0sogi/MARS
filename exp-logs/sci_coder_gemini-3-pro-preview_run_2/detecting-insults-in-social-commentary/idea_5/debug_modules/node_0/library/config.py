import os
import torch
import random
import numpy as np


class Config:
    """
    Configuration class for the Insult Detection task using DeBERTa-v3-Large.
    Implements settings for Self-Training and Adversarial Weight Perturbation (AWP).
    """

    # ====================================================
    # General Settings
    # ====================================================
    seed = 42
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ====================================================
    # Data & Paths
    # ====================================================
    # Read-only input directories
    input_dir = "./input"
    metadata_dir = "./metadata"

    # Working directory for caching and outputs
    working_dir = "./working/idea_5"
    output_dir = "./working/idea_5"

    # File paths
    train_path = os.path.join(metadata_dir, "train.csv")
    val_path = os.path.join(metadata_dir, "val.csv")
    test_path = os.path.join(metadata_dir, "test.csv")
    submission_path = os.path.join(output_dir, "submission.csv")

    # ====================================================
    # Model Architecture
    # ====================================================
    model_name = "microsoft/deberta-v3-large"

    # Tokenizer settings
    max_len = 512  # Sufficient for comment length distribution

    # Head settings
    num_classes = 1
    dropout_rates = [0.1, 0.2, 0.3, 0.4, 0.5]  # For Multi-Sample Dropout
    fc_dropout = 0.2

    # ====================================================
    # Training Hyperparameters
    # ====================================================
    # Batch sizes (tuned for A100 40GB)
    train_batch_size = 4
    valid_batch_size = 16
    gradient_accumulation_steps = 4  # Effective batch size = 16

    # Optimization
    max_grad_norm = 1000
    weight_decay = 0.01
    scheduler = "cosine"
    warmup_ratio = 0.1

    # Layer-wise Learning Rate Decay (LLRD)
    lr = 1e-5
    min_lr = 1e-7
    llrd_decay = 0.9

    # Cross-Validation
    n_folds = 5

    # ====================================================
    # Stage 1: Teacher Training (Standard Fine-Tuning)
    # ====================================================
    stage1_epochs = 3

    # ====================================================
    # Pseudo-Labeling
    # ====================================================
    pseudo_label_threshold = 0.90  # Confidence threshold for hard labels

    # ====================================================
    # Stage 2: Student Training (AWP)
    # ====================================================
    stage2_epochs = 5

    # Adversarial Weight Perturbation (AWP) settings
    use_awp = True
    awp_start_epoch = 1  # Start AWP after 1 epoch of stabilization
    awp_eps = 1e-2  # Epsilon for weight perturbation
    awp_lr = 1e-4  # Learning rate for AWP maximization step

    # ====================================================
    # Debugging & Validation
    # ====================================================
    debug = False  # Set to True to run on a small subset
    debug_sample_size = 100
    print_freq = 50  # Logging frequency

    @classmethod
    def setup(cls):
        """
        Sets up the environment: creates directories and sets seeds.
        """
        # Create working directory
        os.makedirs(cls.working_dir, exist_ok=True)

        # Set seeds for reproducibility
        random.seed(cls.seed)
        np.random.seed(cls.seed)
        torch.manual_seed(cls.seed)
        torch.cuda.manual_seed(cls.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Suppress tokenizers parallelism warning
        os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Automatically run setup when module is imported to ensure consistency
Config.setup()
