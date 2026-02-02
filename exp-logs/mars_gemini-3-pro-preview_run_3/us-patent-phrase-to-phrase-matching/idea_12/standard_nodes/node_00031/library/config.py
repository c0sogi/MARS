import os
import torch


class CFG:
    """
    Configuration class for the Stratified Dynamic-Layer Ensemble with Metric-Aligned Optimization.
    """

    # ====================================================
    # General Settings
    # ====================================================
    debug = False  # Set to True to run with a small subset of data for debugging
    debug_sample_size = 1000  # Number of samples to use in debug mode
    num_workers = 4
    seed = 42
    print_freq = 50

    # ====================================================
    # Paths
    # ====================================================
    # Root directories
    input_root = "./input"
    metadata_root = "./metadata"
    working_dir = "./working/idea_12"

    # Ensure working directory exists
    os.makedirs(working_dir, exist_ok=True)

    # Data paths
    train_path = os.path.join(metadata_root, "train.csv")
    val_path = os.path.join(metadata_root, "val.csv")
    test_path = os.path.join(metadata_root, "test.csv")
    sample_submission_path = os.path.join(input_root, "sample_submission.csv")

    # Caching paths (Parquet format preferred over Pickle)
    context_map_path = os.path.join(working_dir, "context_map.parquet")
    train_cache_path = os.path.join(working_dir, "train_cache.parquet")
    val_cache_path = os.path.join(working_dir, "val_cache.parquet")
    test_cache_path = os.path.join(working_dir, "test_cache.parquet")

    # Output paths
    output_dir = working_dir
    submission_path = "./submission/submission.csv"

    # ====================================================
    # Model Architecture
    # ====================================================
    model_name = "microsoft/deberta-v3-large"
    gradient_checkpointing = True  # Essential for Large models on GPU

    # Input dimensions
    max_len = 140  # Sufficient for Anchor + Target + Expanded Context
    target_size = 1

    # Dynamic Layer Mixing
    use_mix_layers = True  # Enable weighted scalar mixing of all hidden layers

    # Dropout
    fc_dropout = 0.2

    # ====================================================
    # Training Hyperparameters
    # ====================================================
    n_fold = 5
    trn_fold = [0, 1, 2, 3, 4]  # Folds to train

    epochs = 5
    batch_size = 16  # Tuned for A100 40GB
    gradient_accumulation_steps = 1
    max_grad_norm = 1000  # High tolerance, AWP handles regularization

    # Optimizer (AdamW)
    encoder_lr = 2e-5  # Lower learning rate for the backbone
    head_lr = 1e-4  # Higher learning rate for the custom head
    weight_decay = 0.01
    eps = 1e-6
    betas = (0.9, 0.999)

    # Scheduler (Cosine with Warmup)
    scheduler_type = "cosine"
    warmup_ratio = 0.1
    num_cycles = 0.5

    # Layer-wise Learning Rate Decay (LLRD)
    llrd = True
    layer_decay = 0.9  # Decay rate for lower layers

    # ====================================================
    # Loss Function (Hybrid Pearson Loss)
    # ====================================================
    # L_Total = L_MSE + lambda_ce * L_CE + lambda_pearson * (1 - rho)
    loss_config = {
        "mse_weight": 1.0,
        "ce_weight": 0.2,  # Weight for auxiliary classification loss
        "pearson_weight": 0.5,  # Weight for direct correlation optimization
        "ce_bins": 10,  # Number of bins for classification head if used
    }

    # ====================================================
    # Advanced Regularization
    # ====================================================
    # Adversarial Weight Perturbation (AWP)
    awp = True
    awp_eps = 1e-2
    awp_lr = 1e-4
    awp_start_epoch = (
        1  # Start AWP after the first epoch to stabilize initial convergence
    )

    # Exponential Moving Average (EMA)
    ema = True
    ema_decay = 0.999
    ema_start_epoch = 0

    # ====================================================
    # Hardware
    # ====================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fp16 = True  # Use Mixed Precision
