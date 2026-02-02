import os
import torch


class Config:
    """
    Centralized configuration for the Stratified Ensemble with Weighted Layer Pooling
    and Full-Convergence Optimization (Idea 13).
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    seed = 42
    debug = False
    debug_sample_size = 100
    num_workers = 4
    print_freq = 50

    # =========================================================================
    # Paths
    # =========================================================================
    input_dir = "./input"
    metadata_dir = "./metadata"
    working_dir = "./working/idea_13"
    output_dir = os.path.join(working_dir, "output")

    # Metadata paths (Stratified Splits)
    train_path = os.path.join(metadata_dir, "train.csv")
    val_path = os.path.join(metadata_dir, "val.csv")
    test_path = os.path.join(metadata_dir, "test.csv")
    submission_path = os.path.join(working_dir, "submission.csv")

    # Context Data (to be generated/cached)
    cpc_context_path = os.path.join(working_dir, "cpc_context_map.parquet")

    # =========================================================================
    # Model Architecture
    # =========================================================================
    model_name = "microsoft/deberta-v3-large"
    max_len = 140  # Accommodate Anchor + Target + Hierarchical Context

    # Pooling & Head
    pool_type = "weighted_layer"  # Weighted Layer Pooling (Scalar Mixing)
    num_msd = 5  # Multi-Sample Dropout runs
    msd_dropout = 0.1  # Dropout rate for MSD
    fc_dropout = 0.0  # Dropout for fully connected layers

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    n_fold = 5
    trn_folds = [0, 1, 2, 3, 4]
    epochs = 5  # Full convergence protocol

    # Batch Size & Gradient Accumulation
    # A100 40GB: Batch 8 is safe with AWP (which increases memory usage)
    train_batch_size = 8
    valid_batch_size = 16
    gradient_accumulation_steps = 1
    max_grad_norm = 1000

    # Optimization (AdamW)
    encoder_lr = 2e-5
    head_lr = 1e-4
    weight_decay = 0.01
    eps = 1e-6
    betas = (0.9, 0.999)

    # Scheduler
    scheduler_type = "cosine"
    num_warmup_steps = 0
    warmup_ratio = 0.1
    num_cycles = 0.5

    # Layer-wise Learning Rate Decay (LLRD)
    llrd_rate = 0.9

    # =========================================================================
    # Regularization (AWP & EMA)
    # =========================================================================
    # Adversarial Weight Perturbation
    use_awp = True
    awp_start_epoch = 1.0  # Start AWP after 1st epoch
    awp_eps = 1e-4
    awp_lr = 1e-4

    # Exponential Moving Average
    use_ema = True
    ema_decay = 0.999
    ema_start_epoch = 0

    # =========================================================================
    # Loss Function (Hybrid Pearson Loss)
    # =========================================================================
    # L_Total = L_MSE + lambda1 * L_CE + lambda2 * (1 - Pearson)
    loss_mse_weight = 1.0
    loss_ce_weight = 0.5
    loss_pearson_weight = 1.0

    # =========================================================================
    # Device
    # =========================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, **kwargs):
        """
        Initialize Config with optional overrides.
        """
        # Update attributes with passed keyword arguments
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                # Warn or set new attribute? Setting new is flexible.
                setattr(self, k, v)

        # Ensure working directories exist
        os.makedirs(self.working_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

    def to_dict(self):
        """Return configuration as a dictionary."""
        return {
            k: v
            for k, v in self.__class__.__dict__.items()
            if not k.startswith("__") and not callable(v)
        }
