import os
import torch


class Config:
    # ====================================================
    # General Settings
    # ====================================================
    seed = 42
    debug = False  # Set to True for fast debugging runs
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ====================================================
    # Directories & Paths
    # ====================================================
    input_dir = "./input"
    metadata_dir = "./metadata"
    working_dir = "./working/idea_8"
    submission_dir = "./submission"

    # Input Files (from Metadata for splits, Input for raw)
    train_file = os.path.join(metadata_dir, "train.csv")
    val_file = os.path.join(metadata_dir, "val.csv")
    test_file = os.path.join(metadata_dir, "test.csv")
    sample_submission_file = os.path.join(input_dir, "sample_submission.csv")

    # Cache Files (Parquet/NPY for efficient loading)
    context_map_path = os.path.join(working_dir, "context_map.parquet")
    train_cache_path = os.path.join(working_dir, "train_cache.parquet")
    val_cache_path = os.path.join(working_dir, "val_cache.parquet")
    test_cache_path = os.path.join(working_dir, "test_cache.parquet")

    # Output
    submission_path = os.path.join(submission_dir, "submission.csv")
    model_output_dir = os.path.join(working_dir, "models")

    # ====================================================
    # Model Architecture
    # ====================================================
    model_name = "microsoft/deberta-v3-large"
    max_length = 140  # Sufficient for Anchor + Target + Hierarchical Context

    # Custom Architecture Flags
    use_weighted_layer_pooling = True  # Scalar mixing of all layers
    use_attention_pooling = True  # Attention-based aggregation
    use_multi_sample_dropout = True  # Multiple dropout masks in head

    # Head Settings
    dropout_rate = 0.1
    multi_sample_dropout_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
    num_classes = 1  # Regression output

    # ====================================================
    # Training Hyperparameters
    # ====================================================
    n_folds = 5
    epochs = 5
    batch_size = 16
    gradient_accumulation_steps = 1
    max_grad_norm = (
        1000.0  # High value to allow AWP to manage gradients, or standard clipping
    )

    # Optimizer (AdamW)
    encoder_lr = 1e-5  # Lower LR for backbone
    head_lr = 5e-5  # Higher LR for custom heads
    weight_decay = 0.01
    eps = 1e-6
    betas = (0.9, 0.999)

    # Layer-wise Learning Rate Decay (LLRD)
    llrd_decay = 0.9  # Decay rate for deeper layers

    # Scheduler
    scheduler_type = "cosine"
    warmup_ratio = 0.1
    num_cycles = 0.5

    # ====================================================
    # Regularization & Loss
    # ====================================================
    # Adversarial Weight Perturbation (AWP)
    use_awp = True
    awp_lr = 1e-4
    awp_eps = 1e-4
    awp_start_epoch = 1.0  # Start AWP after the first epoch

    # Hybrid Pearson Loss Weights
    # Loss = mse_weight*MSE + ce_weight*CE + pearson_weight*(1 - Pearson)
    mse_weight = 1.0
    ce_weight = 0.5
    pearson_weight = 0.5

    # Auxiliary Classification
    aux_num_classes = 5  # 0.0, 0.25, 0.5, 0.75, 1.0

    # ====================================================
    # Logging & Validation
    # ====================================================
    print_freq = 100
    eval_freq = 1  # Evaluate every epoch

    def __init__(self):
        """Initialize configuration and create necessary directories."""
        os.makedirs(self.working_dir, exist_ok=True)
        os.makedirs(self.submission_dir, exist_ok=True)
        os.makedirs(self.model_output_dir, exist_ok=True)
