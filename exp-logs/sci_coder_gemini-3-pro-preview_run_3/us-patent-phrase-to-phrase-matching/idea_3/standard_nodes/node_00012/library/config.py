import os
import torch


class Config:
    """
    Configuration class for the Phrase Matching task.
    Centralizes all hyperparameters, file paths, and model settings.
    """

    # ==========================================
    # General Settings
    # ==========================================
    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Use available CPUs for data loading, capping at a reasonable number
    num_workers = 4

    # ==========================================
    # Data Paths
    # ==========================================
    # Using metadata generated in ./metadata as per instructions
    train_metadata_path = "./metadata/train.csv"
    val_metadata_path = "./metadata/val.csv"
    test_metadata_path = "./metadata/test.csv"

    # Reference to sample submission for formatting
    sample_submission_path = "./input/sample_submission.csv"

    # ==========================================
    # Output Directories & Files
    # ==========================================
    # Working directory for caching and model checkpoints
    working_dir = "./working/idea_3/"
    model_path = os.path.join(working_dir, "model.pth")

    # Submission directory
    submission_dir = "./submission/"
    submission_path = os.path.join(submission_dir, "submission.csv")

    # ==========================================
    # Model Architecture
    # ==========================================
    model_name = "microsoft/deberta-v3-large"

    # Max Sequence Length:
    # Context + Anchor + Target. EDA shows texts are short, but we use 130
    # to comfortably accommodate special tokens and the longest combinations.
    max_length = 130

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    # Toggle for debugging (runs on a subset of data if True)
    debug = False

    # Training duration
    epochs = 5

    # AWP Hyperparameters
    awp_start_epoch = 1  # Start AWP after the first epoch
    awp_lr = 1e-5  # Perturbation learning rate
    awp_eps = 1e-3  # Epsilon

    # Batch sizes (Adjusted for A100 40GB)
    train_batch_size = 16
    valid_batch_size = 32

    # Optimizer settings
    learning_rate = 2e-5
    weight_decay = 0.01
    eps = 1e-6
    max_grad_norm = 1.0

    # Layer-wise Learning Rate Decay (LLRD)
    # Lower layers get smaller LR to preserve pre-trained knowledge
    llrd_decay = 0.9

    # Scheduler settings
    scheduler_type = "cosine"
    warmup_ratio = 0.1

    def __init__(self):
        """
        Initialize configuration and ensure necessary directories exist.
        """
        os.makedirs(self.working_dir, exist_ok=True)
        os.makedirs(self.submission_dir, exist_ok=True)
