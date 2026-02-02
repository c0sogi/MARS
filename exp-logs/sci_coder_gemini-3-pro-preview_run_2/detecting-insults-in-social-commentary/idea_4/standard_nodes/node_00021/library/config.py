import os
import torch


class Config:
    # =========================================================================
    # General & System Settings
    # =========================================================================
    seed = 42
    debug = False  # Set to True to run on a small subset for debugging
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = True  # Enable Mixed Precision Training

    # =========================================================================
    # File Paths & Directories
    # =========================================================================
    input_dir = "./input"
    metadata_dir = "./metadata"
    working_dir = "./working/idea_4"
    submission_dir = "./submission"

    # Ensure necessary directories exist
    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(submission_dir, exist_ok=True)

    # Data File Paths
    train_path = os.path.join(metadata_dir, "train.csv")
    val_path = os.path.join(metadata_dir, "val.csv")
    test_path = os.path.join(metadata_dir, "test.csv")
    submission_path = os.path.join(submission_dir, "submission.csv")

    # =========================================================================
    # Model Architecture
    # =========================================================================
    # Heterogeneous Ensemble: List of backbones to train
    model_backbones = ["microsoft/deberta-v3-large", "roberta-large"]

    # Classification Head Settings
    num_classes = 1
    dropout_rate = 0.1
    num_msd = 5  # Number of Multi-Sample Dropout branches

    # =========================================================================
    # Tokenization
    # =========================================================================
    max_len = 128  # Maximum sequence length

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    n_folds = 5
    epochs = 4
    batch_size = 16  # Adjusted for A100 40GB

    # Optimizer Settings (AdamW)
    lr = 1e-5
    weight_decay = 0.01
    max_grad_norm = 1000.0

    # Scheduler Settings
    scheduler_type = "cosine"
    warmup_ratio = 0.1

    # Layer-wise Learning Rate Decay (LLRD)
    llrd_decay = 0.9  # Decay rate for lower layers

    # =========================================================================
    # Adversarial Weight Perturbation (AWP) Settings
    # =========================================================================
    use_awp = True
    awp_lr = 1e-4
    awp_eps = 1e-4
    awp_start_epoch = 1  # Epoch to start AWP training
