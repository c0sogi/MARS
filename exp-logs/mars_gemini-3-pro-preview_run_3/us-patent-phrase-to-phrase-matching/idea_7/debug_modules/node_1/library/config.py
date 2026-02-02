import os
import torch


class Config:
    """
    Configuration class for Idea 7: Hierarchical Context-Aware Cross-Encoder
    with Multi-Layer Pooling, DAPT, AWP, and EMA.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    seed = 42
    debug = False  # Set to True to run on a small subset for debugging
    debug_sample_size = 100  # Number of samples to use when debug=True
    exp_name = "idea_7"

    # =========================================================================
    # File Paths & Directories
    # =========================================================================
    # Input Data (Read-Only Metadata)
    input_root = "./metadata"
    train_path = os.path.join(input_root, "train.csv")
    val_path = os.path.join(input_root, "val.csv")
    test_path = os.path.join(input_root, "test.csv")
    sample_submission_path = "./input/sample_submission.csv"

    # Working Directory (For intermediate files, caches, models)
    working_dir = os.path.join("./working", exp_name)
    os.makedirs(working_dir, exist_ok=True)

    # Output Submission
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)
    submission_path = os.path.join(submission_dir, "submission.csv")

    # Cache Files (Parquet format for speed and type safety)
    # Used to store processed datasets with hierarchical context expansion
    train_cache_path = os.path.join(working_dir, "train_cache.parquet")
    val_cache_path = os.path.join(working_dir, "val_cache.parquet")
    test_cache_path = os.path.join(working_dir, "test_cache.parquet")
    context_map_cache_path = os.path.join(working_dir, "context_map.parquet")

    # Model Checkpoints
    best_model_path = os.path.join(working_dir, "best_model.pth")

    # =========================================================================
    # Model Architecture
    # =========================================================================
    model_backbone = "microsoft/deberta-v3-large"

    # Input formatting
    # Contexts + Anchor + Target can be lengthy; 133 covers >99% of cases
    max_length = 133

    # Cross-Encoder Heads
    target_size = 1  # Regression score (continuous)
    num_classes = 5  # Classification bins: 0.0, 0.25, 0.5, 0.75, 1.0

    # Multi-Layer Attention Pooling
    pooling_type = "multi_layer_attention"
    pool_layers = 4  # Aggregate hidden states from the last 4 layers

    # Dropout Regularization
    hidden_dropout_prob = 0.1
    attention_probs_dropout_prob = 0.1
    multi_sample_dropout_rate = 0.2
    multi_sample_dropout_num = 5

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    epochs = 5
    train_batch_size = 8  # Fits on A100-40GB with DeBERTa-Large
    valid_batch_size = 16
    gradient_accumulation_steps = 1
    max_grad_norm = 1000.0  # High clipping threshold for AWP stability

    # Optimizer (AdamW)
    encoder_lr = 2e-5  # Lower LR for pre-trained backbone
    decoder_lr = 1e-4  # Higher LR for randomly initialized heads
    weight_decay = 0.01
    eps = 1e-6
    beta1 = 0.9
    beta2 = 0.999

    # Scheduler (Cosine with Warmup)
    scheduler_type = "cosine"
    warmup_ratio = 0.1
    num_cycles = 0.5

    # Layer-wise Learning Rate Decay (LLRD)
    llrd_decay = 0.9  # Decay rate for deeper layers to preserve features

    # =========================================================================
    # Advanced Training Techniques
    # =========================================================================

    # 1. Domain-Adaptive Pre-training (DAPT)
    use_dapt = True
    dapt_model_path = os.path.join(working_dir, "dapt_model")
    dapt_epochs = 3
    dapt_batch_size = 8
    dapt_mlm_probability = 0.15
    dapt_lr = 2e-5

    # 2. Adversarial Weight Perturbation (AWP)
    use_awp = True
    awp_start_epoch = 1.0  # Enable AWP after the first epoch
    awp_eps = 1e-2  # Perturbation magnitude
    awp_lr = 1e-4  # AWP learning rate

    # 3. Exponential Moving Average (EMA)
    use_ema = True
    ema_decay = 0.999
    ema_start_epoch = 0

    # 4. Composite Loss Function
    # Loss = MSE + lambda_ce * CE + lambda_pearson * (1 - Pearson)
    loss_mse_weight = 1.0
    loss_ce_weight = 0.5
    loss_pearson_weight = 1.0

    # =========================================================================
    # System & Logging
    # =========================================================================
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_freq = 100  # Logging frequency in steps
