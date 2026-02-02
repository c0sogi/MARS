import os
import torch
import random
import numpy as np


class Config:
    """
    Configuration class for the DeBERTa-v3-Large Cross-Encoder solution.
    Defines paths, model hyperparameters, and training settings.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    seed = 42
    debug = False  # Set to True to run on a small subset for rapid testing
    debug_sample_size = 200  # Number of samples to use when debug=True
    num_workers = 4  # Number of dataloader workers
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================================================================
    # File Paths
    # =========================================================================
    # Input Directories (Read-Only)
    input_dir = "./input"
    metadata_dir = "./metadata"

    # Data Files
    cpc_path = os.path.join(input_dir, "description.md")
    train_path = os.path.join(metadata_dir, "train.csv")
    val_path = os.path.join(metadata_dir, "val.csv")
    test_path = os.path.join(metadata_dir, "test.csv")
    sample_submission_path = os.path.join(input_dir, "sample_submission.csv")

    # Working Directories (Writeable)
    # Using 'idea_7' as the current experiment version identifier
    working_dir = "./working/idea_7"
    models_dir = os.path.join(working_dir, "models")
    predictions_dir = os.path.join(working_dir, "predictions")
    cache_dir = os.path.join(working_dir, "cache")

    # Submission Output
    submission_dir = "./submission"
    submission_path = os.path.join(submission_dir, "submission.csv")

    # =========================================================================
    # Model Architecture
    # =========================================================================
    model_name = "microsoft/deberta-v3-large"

    # Max Length: Sufficient for Context Description (~50-80 tokens) + Anchor + Target
    max_length = 140

    # Multi-Sample Dropout (MSD) Settings
    # Uses multiple dropout masks in the final layer to smooth the loss landscape
    use_msd = True
    msd_rates = [0.1, 0.2, 0.3, 0.4, 0.5]

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    n_folds = 5
    epochs = 4

    # Batch Size Strategy
    # DeBERTa-Large is memory intensive. We use a smaller physical batch size
    # combined with gradient accumulation to achieve a stable effective batch size.
    batch_size = 8
    gradient_accumulation_steps = 4  # Effective batch size = 32

    # Optimization
    lr = 2e-5  # Base learning rate for the transformer backbone
    head_lr = 1e-4  # Higher learning rate for the regression head
    weight_decay = 0.01
    max_grad_norm = 1000.0  # Gradient clipping

    # Scheduler
    scheduler_type = "cosine"
    warmup_ratio = 0.1
    num_cycles = 0.5

    # Layer-wise Learning Rate Decay (LLRD)
    # Multiplicatively decays LR for lower layers to preserve pre-trained linguistic features
    use_llrd = True
    llrd_decay = 0.9

    # Loss Function
    loss_type = "mse"  # Mean Squared Error aligns best with Pearson correlation

    # Advanced
    use_fp16 = True  # Mixed Precision Training

    @classmethod
    def setup(cls):
        """
        Initializes the environment:
        1. Creates necessary directories.
        2. Sets random seeds for reproducibility.
        """
        # 1. Create Directories
        dirs = [
            cls.working_dir,
            cls.models_dir,
            cls.predictions_dir,
            cls.cache_dir,
            cls.submission_dir,
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)

        # 2. Set Random Seeds
        random.seed(cls.seed)
        np.random.seed(cls.seed)
        torch.manual_seed(cls.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cls.seed)
            # Ensure deterministic behavior
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


# Automatically setup environment on import
Config.setup()
