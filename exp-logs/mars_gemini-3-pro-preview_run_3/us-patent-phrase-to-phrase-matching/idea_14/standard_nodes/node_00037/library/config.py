import os
import torch


class Config:
    """
    Centralized configuration for the Phrase Matching Task.
    Implements the 'Stratified Ensemble with Scalar Layer Mixing' strategy.
    """

    # =========================================================================
    # General Environment Setup
    # =========================================================================
    seed = 42
    debug = False  # Set to True to run on a small subset of data for debugging
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================================================================
    # Directories and File Paths
    # =========================================================================
    input_dir = "./input"
    metadata_dir = "./metadata"
    output_dir = "./working/idea_14"

    # Ensure the working directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Data Paths (using generated metadata)
    train_path = os.path.join(metadata_dir, "train.csv")
    val_path = os.path.join(metadata_dir, "val.csv")
    test_path = os.path.join(metadata_dir, "test.csv")
    sample_submission_path = os.path.join(input_dir, "sample_submission.csv")

    # Cache Paths for processed data
    cpc_context_map_path = os.path.join(output_dir, "cpc_context_map.parquet")
    train_cache_path = os.path.join(output_dir, "train_cache.parquet")
    val_cache_path = os.path.join(output_dir, "val_cache.parquet")
    test_cache_path = os.path.join(output_dir, "test_cache.parquet")

    # Output Paths
    model_save_path = os.path.join(output_dir, "model")
    submission_path = os.path.join(output_dir, "submission.csv")

    # =========================================================================
    # Model Architecture
    # =========================================================================
    model_name = "microsoft/deberta-v3-large"
    max_length = 140  # Sufficient for Anchor + Target + Hierarchical Context

    # Scalar Layer Mixing Settings
    use_scalar_mixing = True

    # Dropout
    hidden_dropout_prob = 0.1
    attention_probs_dropout_prob = 0.1

    # =========================================================================
    # Training Hyperparameters (Convergence-Aware Protocol)
    # =========================================================================
    epochs = 5  # Extended training time for convergence
    n_folds = 5

    train_batch_size = 8  # Reduced to 8 to fit in 16GB VRAM with Large model
    valid_batch_size = 32

    # Optimizer (AdamW)
    learning_rate = 2e-5
    weight_decay = 0.01
    adam_epsilon = 1e-6
    adam_betas = (0.9, 0.999)
    max_grad_norm = 1000.0

    # Scheduler
    scheduler_type = "cosine"
    warmup_ratio = 0.1

    # Layer-wise Learning Rate Decay (LLRD)
    llrd_decay = 0.9
    head_lr_scale = 5.0  # Higher LR for the task-specific head

    # =========================================================================
    # Loss Function (Hybrid Pearson Loss)
    # =========================================================================
    # L_Total = L_MSE + lambda_ce * L_CE + lambda_pearson * (1 - rho)
    loss_mse_weight = 1.0
    loss_ce_weight = 0.5
    loss_pearson_weight = 1.0

    # Auxiliary Classification Head
    num_aux_classes = 5  # Bins: 0.0, 0.25, 0.5, 0.75, 1.0

    # =========================================================================
    # Regularization & Stabilization
    # =========================================================================
    # Adversarial Weight Perturbation (AWP)
    use_awp = True
    awp_lr = 1e-4
    awp_eps = 1e-4
    awp_start_epoch = 2  # Start AWP after the model has stabilized (Epoch 2)

    # Exponential Moving Average (EMA)
    use_ema = True
    ema_decay = 0.999

    # =========================================================================
    # Logging & Inference
    # =========================================================================
    print_freq = 50
    inference_batch_size = 32
