import os
import torch


class Config:
    """
    Centralized configuration for the QA Hindi/Tamil task.
    Implements settings for:
    - Model: XLM-Roberta-Large with Multi-Sample Dropout and Layer Re-initialization.
    - Data: Sliding window with negative sampling (2:1 ratio).
    - Training: Adversarial training (FGM), Differential Learning Rates, and Seed Ensemble.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    project_name = "qa_hindi_tamil"
    idea_name = "idea_13"
    verbose = True
    seed = 42  # Base seed (used for data splitting/sampling if needed)

    # Hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 4  # Number of dataloader workers

    # =========================================================================
    # File Paths
    # =========================================================================
    # Input Data (Metadata)
    train_meta_path = "./metadata/train.csv"
    val_meta_path = "./metadata/val.csv"
    test_meta_path = "./metadata/test.csv"

    # Working Directories
    working_dir = f"./working/{idea_name}"
    output_dir = os.path.join(working_dir, "output")
    cache_dir = os.path.join(working_dir, "cache")
    submission_dir = os.path.join(working_dir, "submission")

    # Output Files
    submission_path = os.path.join(submission_dir, "submission.csv")

    # Cache Files (Parquet)
    # We use a combined training cache since we merge train and val
    train_features_file = os.path.join(cache_dir, "train_features.parquet")
    test_features_file = os.path.join(cache_dir, "test_features.parquet")

    # =========================================================================
    # Model Architecture
    # =========================================================================
    model_name = "xlm-roberta-large"
    hidden_size = 1024  # Hidden size for XLM-R Large
    num_labels = 2  # Start and End logits

    # Structural Innovations
    reinit_layers = 1  # Re-initialize the top 1 encoder layer (Layer 23)

    # Multi-Sample Dropout
    use_multi_sample_dropout = True
    multi_sample_dropout_num = 5
    dropout_rate = 0.1  # Base dropout rate

    # =========================================================================
    # Data Processing
    # =========================================================================
    max_length = 384
    doc_stride = 128

    # Negative Sampling Strategy
    # Retain 100% Positive, Downsample Negative to maintain 2:1 ratio
    negative_positive_ratio = 2.0

    # Data Usage
    # Concatenate train.csv and val.csv into a single training set
    use_full_train_data = True

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    epochs = 3
    train_batch_size = 8  # Adjusted for A100 40GB VRAM
    eval_batch_size = 16

    # Optimization
    lr_backbone = 1e-5
    lr_head = 5e-5
    weight_decay = 0.01  # Applied to ALL parameters (including bias/LayerNorm)
    max_grad_norm = 1.0

    # Loss Weighting
    # Total Loss = Span_Loss + relevance_loss_weight * Relevance_Loss
    relevance_loss_weight = 0.5

    # Adversarial Training (FGM)
    use_fgm = True
    fgm_epsilon = 1.0

    # =========================================================================
    # Ensemble Strategy
    # =========================================================================
    # Train 3 independent models with different seeds
    seeds = [42, 43, 44]

    @classmethod
    def setup(cls):
        """
        Creates necessary directories for the experiment.
        """
        os.makedirs(cls.working_dir, exist_ok=True)
        os.makedirs(cls.output_dir, exist_ok=True)
        os.makedirs(cls.cache_dir, exist_ok=True)
        os.makedirs(cls.submission_dir, exist_ok=True)


# Automatically create directories when config is imported/used
Config.setup()
