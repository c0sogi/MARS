import os
import torch


class CFG:
    """
    Configuration class for the Two-Stage Stratified Ensemble with Dynamic Layer Mixing.
    Centralizes all hyperparameters, file paths, and model settings.
    """

    # ====================================================
    # General Settings
    # ====================================================
    wandb = False
    project = "phrase-matching-idea-11"
    model_name = "microsoft/deberta-v3-large"
    seed = 42
    debug = False  # Set to True to run on a small subset for debugging
    print_freq = 100
    num_workers = 4

    # ====================================================
    # Paths
    # ====================================================
    # Input directories
    input_dir = "./input"
    metadata_dir = "./metadata"

    # Metadata files (Stratified Splits)
    train_path = os.path.join(metadata_dir, "train.csv")
    val_path = os.path.join(metadata_dir, "val.csv")
    test_path = os.path.join(metadata_dir, "test.csv")
    sample_submission_path = os.path.join(input_dir, "sample_submission.csv")

    # Output directory for caching data, saving models, and submissions
    output_dir = "./working/idea_11/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Cache file paths
    context_map_path = os.path.join(output_dir, "context_map.parquet")

    # ====================================================
    # Data & Tokenizer
    # ====================================================
    # Max length: Context (approx 50-80) + Anchor (approx 10-20) + Target (approx 10-20) + Special Tokens
    # 175 provides a safe buffer for the hierarchical context expansion.
    max_len = 175

    # ====================================================
    # Training Hyperparameters
    # ====================================================
    n_fold = 5
    trn_fold = [0, 1, 2, 3, 4]  # Folds to train
    epochs = 5

    # Batch Sizes (Tuned for A100 40GB)
    train_batch_size = 16
    valid_batch_size = 32

    # Optimization
    encoder_lr = 2e-5  # Lower LR for the pre-trained backbone
    head_lr = 1e-4  # Higher LR for the custom heads and mixing weights
    min_lr = 1e-6
    weight_decay = 0.01
    gradient_checkpointing = True
    max_grad_norm = 1000

    # Scheduler
    scheduler = "cosine"  # Options: ['linear', 'cosine']
    batch_scheduler = True  # Step scheduler every batch
    num_cycles = 0.5
    warmup_ratio = 0.1

    # ====================================================
    # Model Architecture Specifics
    # ====================================================
    fc_dropout = 0.2
    target_size = 1  # Regression output
    num_classes = 5  # Classification output (0.0, 0.25, 0.5, 0.75, 1.0)

    # ====================================================
    # Advanced Strategies (Idea 11)
    # ====================================================

    # --- Two-Stage Warmup ---
    # Stage 1: Freeze backbone, train mixing weights/heads only.
    # Stage 2: Unfreeze backbone, train full model with LLRD.
    warmup_epochs = 1

    # --- Layer-wise Learning Rate Decay (LLRD) ---
    # Decays LR for lower layers to preserve pre-trained features.
    llrd_decay = 0.9

    # --- Adversarial Weight Perturbation (AWP) ---
    # Perturbs weights to flatten the loss landscape.
    awp = True
    awp_lr = 1e-4
    awp_eps = 1e-4
    awp_start_epoch = 2  # Start AWP after initial stabilization

    # --- Exponential Moving Average (EMA) ---
    # Maintains a moving average of weights for robust inference.
    ema = True
    ema_decay = 0.999

    # --- Hybrid Pearson Loss Weights ---
    # L_Total = L_MSE + lambda_1 * L_CE + lambda_2 * (1 - Pearson)
    loss_weights = {"mse": 1.0, "ce": 0.5, "pearson": 1.0}

    # ====================================================
    # System
    # ====================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
