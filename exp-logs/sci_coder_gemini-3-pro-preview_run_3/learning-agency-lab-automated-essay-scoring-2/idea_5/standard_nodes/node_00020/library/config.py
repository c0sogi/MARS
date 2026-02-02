import os
import torch


class Config:
    """
    Central configuration for the Essay Scoring pipeline using DeBERTa-v3-Large
    and LightGBM Stacking with Adversarial Regularization.
    """

    # -------------------------------------------------------------------------
    # General Configuration
    # -------------------------------------------------------------------------
    seed = 42
    debug = False  # Set to True for fast debugging with a subset of data
    debug_subset_size = 100
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------------------
    # Directory Paths
    # -------------------------------------------------------------------------
    # Input (Read-Only)
    input_dir = "./input"
    metadata_dir = "./metadata"

    # Working Directory (Write Access)
    working_dir = "./working/idea_5"

    # Sub-directories
    output_dir = os.path.join(working_dir, "output")
    checkpoint_dir = os.path.join(working_dir, "checkpoints")
    cache_dir = os.path.join(working_dir, "cache")
    submission_dir = "./submission"

    # Data Source Paths
    # Note: Training will combine train_metadata and val_metadata for 5-fold CV
    train_metadata_path = os.path.join(metadata_dir, "train_metadata.csv")
    val_metadata_path = os.path.join(metadata_dir, "val_metadata.csv")
    test_metadata_path = os.path.join(metadata_dir, "test_metadata.csv")
    sample_submission_path = os.path.join(input_dir, "sample_submission.csv")

    # -------------------------------------------------------------------------
    # Model Architecture
    # -------------------------------------------------------------------------
    model_name = "microsoft/deberta-v3-large"
    tokenizer_path = "microsoft/deberta-v3-large"

    # Input Processing
    max_length = 512
    stride = 256  # For sliding window approach

    # Pooling Strategy: "mean", "max", "attention", "cls"
    pooling_type = "attention"

    # Memory Optimization
    gradient_checkpointing = True

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    n_folds = 5
    epochs = 4

    # Batch Sizes (Adjusted for A100 40GB)
    train_batch_size = 4
    valid_batch_size = 8
    gradient_accumulation_steps = 1

    # Optimizer & Scheduler
    learning_rate = 1e-5
    weight_decay = 0.01
    max_grad_norm = 10.0
    scheduler_type = "cosine"
    warmup_ratio = 0.1

    # Layer-wise Learning Rate Decay (LLRD)
    use_llrd = True
    llrd_decay = 0.9

    # Mixed Precision Training
    use_amp = True

    # -------------------------------------------------------------------------
    # Adversarial Weight Perturbation (AWP)
    # -------------------------------------------------------------------------
    use_awp = True
    awp_start_epoch = 1  # Start AWP after the first epoch (0-indexed: epoch 1)
    awp_lr = 1e-4
    awp_eps = 1e-2

    # -------------------------------------------------------------------------
    # Feature Engineering
    # -------------------------------------------------------------------------
    # Explicit meta-features to concatenate with OOF embeddings
    meta_features = ["char_count", "word_count", "sentence_count", "unique_word_ratio"]

    # -------------------------------------------------------------------------
    # Stacking Model (LightGBM)
    # -------------------------------------------------------------------------
    lgbm_params = {
        "n_estimators": 2000,
        "learning_rate": 0.005,
        "metric": "rmse",
        "random_state": seed,
        "n_jobs": -1,
        "feature_fraction": 0.7,  # < 1.0 to prevent over-reliance on meta-features
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbosity": -1,
        "objective": "regression",
    }

    def __init__(self):
        """
        Initialize the configuration and ensure necessary directories exist.
        """
        # Create working directories
        for d in [
            self.working_dir,
            self.output_dir,
            self.checkpoint_dir,
            self.cache_dir,
            self.submission_dir,
        ]:
            os.makedirs(d, exist_ok=True)
