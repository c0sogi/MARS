import os
import torch


class Config:
    """
    Centralized configuration for the Phrase Similarity scoring task.
    Implements settings for DeBERTa-v3-Large, LLRD, AWP, and Stratified Group K-Fold.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 4
    debug = False  # Set to True to run on a small subset for debugging

    # =========================================================================
    # Data Paths & Settings
    # =========================================================================
    # Metadata paths (pre-generated splits)
    train_path = "./metadata/train.csv"
    val_path = "./metadata/val.csv"
    test_path = "./metadata/test.csv"

    # Raw input for context expansion
    cpc_path = "./input/description.md"

    # Output directory for checkpoints and predictions
    output_dir = "./working/idea_4"

    # Data Processing
    max_length = (
        130  # Anchor + Target + Context Description usually fits in < 100 tokens
    )
    target_col = "score"

    # Cross-Validation
    n_folds = 5
    group_col = "anchor"  # For Stratified Group K-Fold

    # =========================================================================
    # Model Architecture
    # =========================================================================
    model_name = "microsoft/deberta-v3-large"
    num_labels = 1  # Regression (0-1) treated as single logit for BCE
    dropout = 0.0
    attention_dropout = 0.0

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    epochs = 4
    train_batch_size = 8  # Adjusted for A100 40GB with DeBERTa-Large + AWP overhead
    valid_batch_size = 16

    # Optimizer (AdamW)
    learning_rate = 2e-5
    weight_decay = 0.01
    eps = 1e-6
    betas = (0.9, 0.999)
    max_grad_norm = 1000.0

    # Scheduler (Cosine with Warmup)
    scheduler_type = "cosine"
    warmup_ratio = 0.1
    num_cycles = 0.5

    # Loss Function
    loss_fn = "BCEWithLogitsLoss"

    # =========================================================================
    # Advanced Optimization: LLRD (Layer-wise Learning Rate Decay)
    # =========================================================================
    # Decay rate for learning rates from top to bottom layers
    # lr_layer_i = lr_base * (layer_decay ^ (num_layers - i))
    layer_decay = 0.9

    # =========================================================================
    # Advanced Optimization: AWP (Adversarial Weight Perturbation)
    # =========================================================================
    use_awp = True
    awp_start_epoch = 1.0  # Start AWP after the 1st epoch (warmup)
    awp_eps = 1e-4  # Epsilon for adversarial perturbation
    awp_lr = 1e-4  # Learning rate for AWP step

    # =========================================================================
    # Logging & Saving
    # =========================================================================
    print_freq = 100
    save_best_only = True

    @classmethod
    def create_output_dir(cls):
        """Ensures the output directory exists."""
        os.makedirs(cls.output_dir, exist_ok=True)


# Ensure directory exists upon import/usage
Config.create_output_dir()
