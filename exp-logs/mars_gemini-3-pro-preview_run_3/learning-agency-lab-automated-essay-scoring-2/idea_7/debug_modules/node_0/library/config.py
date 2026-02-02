import os
import torch


class Config:
    """
    Centralized configuration for the Essay Scoring pipeline.
    Implements settings for DeBERTa-v3-large, AWP, LLRD, and LightGBM stacking.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    seed = 42
    debug = False  # Set to True to run on a small subset for debugging
    num_workers = 4

    # =========================================================================
    # Data Configuration
    # =========================================================================
    n_folds = 5
    max_length = 512
    stride = 128  # Stride for sliding window inference

    # =========================================================================
    # Model Architecture
    # =========================================================================
    model_name = "microsoft/deberta-v3-large"
    num_classes = 1  # Regression target
    pooling = "attention"  # Options: "mean", "max", "attention", "cls"

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    epochs = 4
    batch_size = 4  # Adjusted for A100 40GB with Large model
    gradient_accumulation_steps = 2

    # Optimizer & Scheduler
    lr = 1e-5
    weight_decay = 0.01
    max_grad_norm = 10.0
    scheduler_type = "cosine"
    warmup_ratio = 0.1

    # Advanced Training Techniques
    use_gradient_checkpointing = True
    use_mixed_precision = True  # Uses torch.amp

    # Layer-wise Learning Rate Decay (LLRD)
    use_llrd = True
    llrd_decay = 0.9

    # Adversarial Weight Perturbation (AWP)
    use_awp = True
    awp_lr = 1e-4
    awp_eps = 1e-2
    awp_start_epoch = 1  # Start AWP after the first epoch

    # =========================================================================
    # Stacking Head (LightGBM)
    # =========================================================================
    # Feature fraction < 1.0 to force usage of semantic embeddings over meta-features
    lgbm_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.005,
        "num_leaves": 31,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbosity": -1,
        "n_estimators": 5000,
        "early_stopping_rounds": 100,
        "random_state": seed,
        "n_jobs": -1,
    }

    # =========================================================================
    # File Paths & Directories
    # =========================================================================
    # Input (Read-Only)
    input_dir = "./input"
    metadata_dir = "./metadata"
    train_metadata_path = os.path.join(metadata_dir, "train_metadata.csv")
    val_metadata_path = os.path.join(metadata_dir, "val_metadata.csv")
    test_metadata_path = os.path.join(metadata_dir, "test_metadata.csv")

    # Working Directory (Write Access)
    working_dir = "./working/idea_7"
    output_dir = os.path.join(working_dir, "output")
    model_dir = os.path.join(working_dir, "checkpoints")
    cache_dir = os.path.join(working_dir, "cache")

    # Submission
    submission_dir = "./submission"
    submission_path = os.path.join(submission_dir, "submission.csv")

    @classmethod
    def setup(cls):
        """
        Creates necessary output directories.
        """
        os.makedirs(cls.output_dir, exist_ok=True)
        os.makedirs(cls.model_dir, exist_ok=True)
        os.makedirs(cls.cache_dir, exist_ok=True)
        os.makedirs(cls.submission_dir, exist_ok=True)

    @property
    def device(self):
        """Returns the appropriate torch device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
