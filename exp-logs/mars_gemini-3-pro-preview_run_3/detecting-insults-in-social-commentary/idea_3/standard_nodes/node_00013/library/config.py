import os
import torch


class Config:
    # ==========================================
    # General Configuration
    # ==========================================
    seed = 42
    debug = False  # Set to True to run with a small subset of data
    debug_sample_size = 100
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==========================================
    # Data Paths
    # ==========================================
    # Using metadata paths as requested
    train_path = "./metadata/train.csv"
    val_path = "./metadata/val.csv"
    test_path = "./metadata/test.csv"

    # Output Directories
    output_dir = "./working/idea_3/"
    submission_dir = "./submission/"
    submission_path = os.path.join(submission_dir, "submission.csv")

    # ==========================================
    # Model Architecture
    # ==========================================
    model_path = "microsoft/deberta-v3-large"
    max_len = 160
    target_col = "Insult"

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    epochs = 4
    train_batch_size = 8  # Adjusted for DeBERTa-Large on 40GB GPU
    valid_batch_size = 16
    learning_rate = 1e-5
    min_lr = 1e-7
    weight_decay = 0.01
    max_grad_norm = 1.0

    # ==========================================
    # Advanced / Regularization
    # ==========================================
    # Layer-wise Learning Rate Decay
    llrd_decay = 0.9

    # Multi-Sample Dropout Rates
    dropout_rates = [0.1, 0.2, 0.3, 0.4, 0.5]

    # Scheduler
    scheduler = "linear"
    warmup_ratio = 0.1


# Ensure directories exist
os.makedirs(Config.output_dir, exist_ok=True)
os.makedirs(Config.submission_dir, exist_ok=True)
