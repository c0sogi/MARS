import os
import torch

# Ensure the working directory for Idea 9 exists
os.makedirs("./working/idea_9/", exist_ok=True)
os.makedirs("./submission/", exist_ok=True)


class Config:
    """
    Configuration class for Idea 9: Robust Stratified Ensemble with Dynamic Layer Aggregation.
    Centralizes all hyperparameters for model architecture, training, and data processing.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    seed = 42
    n_folds = 5
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_freq = 100  # Logging frequency

    # =========================================================================
    # Paths
    # =========================================================================
    working_dir = "./working/idea_9/"
    input_dir = "./input/"
    metadata_dir = "./metadata/"
    submission_path = "./submission/submission.csv"

    # =========================================================================
    # Model Architecture
    # =========================================================================
    model_name = "microsoft/deberta-v3-large"
    max_len = 140  # Accommodates hierarchical context + anchor + target
    hidden_dim = 1024  # Hidden size for Deberta Large
    dropout = 0.0  # Zero dropout often helps regression stability on Transformers

    # Dynamic Layer Aggregation
    use_weighted_layer_pooling = True  # Enable scalar mixing of all layers

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    debug = False  # Set to True to run on a small subset for debugging
    debug_sample_size = 500  # Number of samples to use when debug=True

    epochs = 5
    train_batch_size = 8  # Adjusted for A100 40GB with Deberta Large
    valid_batch_size = 16
    gradient_accumulation_steps = 1
    max_grad_norm = 1000.0  # Gradient clipping

    # Optimization
    learning_rate = 2e-5  # Base learning rate for the backbone
    head_lr = 1e-4  # Higher learning rate for the custom heads
    weight_decay = 0.01

    # Scheduler
    scheduler_type = "cosine"
    warmup_ratio = 0.1

    # Layer-wise Learning Rate Decay (LLRD)
    llrd_decay = 0.9  # Decay factor for layer-wise learning rates

    # =========================================================================
    # Advanced Regularization & Stabilization
    # =========================================================================
    # Adversarial Weight Perturbation (AWP)
    use_awp = True
    awp_lr = 1e-4
    awp_eps = 1e-2
    awp_start_epoch = 1.0  # Start AWP after the first epoch

    # Exponential Moving Average (EMA)
    use_ema = True
    ema_decay = 0.999
    ema_start_epoch = 0

    # =========================================================================
    # Loss Function Configuration
    # =========================================================================
    # Hybrid Loss: L_Total = L_MSE + lambda1 * L_BCE + lambda2 * (1 - Pearson)
    loss_weights = {
        "mse": 1.0,
        "bce": 0.5,  # Auxiliary classification loss weight
        "pearson": 0.5,  # Direct metric optimization weight
    }

    # Auxiliary Classification Targets
    # Scores are discrete: 0.0, 0.25, 0.5, 0.75, 1.0
    num_classification_bins = 5
