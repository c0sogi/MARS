import os
import torch


class CFG:
    # ====================================================
    # General Settings
    # ====================================================
    debug = False  # Set to True to run with a small subset of data for debugging
    debug_sample_size = 500  # Number of samples to use when debug=True
    seed = 42
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_freq = 100

    # ====================================================
    # Data Paths
    # ====================================================
    # Using the stratified metadata generated in previous steps
    train_path = "./metadata/train.csv"
    val_path = "./metadata/val.csv"
    test_path = "./metadata/test.csv"
    sample_submission_path = "./input/sample_submission.csv"

    # Output directory for artifacts (models, logs, cache)
    output_dir = "./working/idea_10/"
    os.makedirs(output_dir, exist_ok=True)

    # ====================================================
    # Model Architecture
    # ====================================================
    model_name = "microsoft/deberta-v3-large"
    max_len = 128  # Length to accommodate anchor + target + hierarchical context
    target_size = 1  # Regression output

    # Dynamic Layer Mixing Settings
    use_dynamic_layer_mixing = True

    # ====================================================
    # Training Hyperparameters
    # ====================================================
    n_folds = 5
    epochs = 5
    train_batch_size = 16  # Tuned for A100-40GB with DeBERTa-Large
    valid_batch_size = 32
    gradient_accumulation_steps = 1
    max_grad_norm = 1000

    # Optimizer
    learning_rate = 2e-5  # Base learning rate
    encoder_lr = 2e-5  # Specific LR for backbone
    head_lr = 1e-4  # Higher LR for the new head layers
    weight_decay = 0.01

    # Scheduler
    scheduler_type = "cosine"
    warmup_ratio = 0.1
    num_cycles = 0.5

    # ====================================================
    # Advanced Techniques
    # ====================================================
    # Layer-wise Learning Rate Decay (LLRD)
    llrd_decay = 0.9

    # Adversarial Weight Perturbation (AWP)
    use_awp = True
    awp_eps = 1e-4
    awp_lr = 1e-4
    awp_start_epoch = 1.0  # Start AWP after the first epoch

    # Exponential Moving Average (EMA)
    use_ema = True
    ema_decay = 0.999
    ema_start_epoch = 0

    # Hybrid Loss Components
    # Loss = MSE + lambda_ce * CE + lambda_pearson * (1 - Pearson)
    lambda_ce = 0.5
    lambda_pearson = 1.0
