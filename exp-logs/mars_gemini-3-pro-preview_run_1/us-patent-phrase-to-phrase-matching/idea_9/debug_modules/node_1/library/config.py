import os
import torch


class Config:
    """
    Configuration class for the U.S. Patent Phrase to Phrase Matching task.
    Centralizes hyperparameters, file paths, and model settings.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 4  # Number of dataloader workers
    debug = False  # Set to True to run on a small subset for debugging

    # =========================================================================
    # File Paths
    # =========================================================================
    input_dir = "./input"
    metadata_dir = "./metadata"
    working_dir = "./working/idea_9"

    # Specific file paths
    train_path = os.path.join(metadata_dir, "train.csv")
    val_path = os.path.join(metadata_dir, "val.csv")
    test_path = os.path.join(metadata_dir, "test.csv")
    cpc_path = os.path.join(input_dir, "description.md")
    sample_submission_path = os.path.join(input_dir, "sample_submission.csv")
    submission_path = os.path.join(
        os.path.dirname(sample_submission_path), "../submission/submission.csv"
    )  # Placeholder logic

    # =========================================================================
    # Model Settings
    # =========================================================================
    model_name = "microsoft/deberta-v3-large"
    max_length = 133  # Sufficient for Context + [SEP] + Anchor + [SEP] + Target
    target_size = 1  # Regression output

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    epochs = 4
    train_batch_size = 8  # Adjusted for A100 40GB with Large model
    valid_batch_size = 16
    gradient_accumulation_steps = 1
    max_grad_norm = 1000.0

    # Optimizer (AdamW)
    lr = 2e-5  # Learning rate for the backbone
    head_lr = 1e-4  # Higher learning rate for the head
    min_lr = 1e-6
    weight_decay = 0.01
    eps = 1e-6
    betas = (0.9, 0.999)

    # Scheduler
    scheduler = "cosine"  # Cosine annealing
    num_warmup_steps = 0  # or percentage of total steps
    batch_scheduler = True

    # Layer-wise Learning Rate Decay (LLRD)
    llrd_decay = 0.9

    # =========================================================================
    # Multi-Sample Dropout (MSD) Head Settings
    # =========================================================================
    fc_dropout = 0.1  # Dropout rate for the fully connected layers
    num_msd = 5  # Number of dropout samples for the head

    # =========================================================================
    # Cross-Validation
    # =========================================================================
    n_fold = 5
    group_col = "anchor"  # Group by anchor to prevent leakage
    stratify_col = "score"  # Stratify by score for balance

    @classmethod
    def setup(cls):
        """
        Creates necessary directories.
        """
        os.makedirs(cls.working_dir, exist_ok=True)
        # Ensure submission directory exists if we were to write there
        os.makedirs(os.path.dirname("./submission/submission.csv"), exist_ok=True)


# Initialize environment
Config.setup()
