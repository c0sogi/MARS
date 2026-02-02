import os
import torch


class Config:
    # ====================================================
    # General Settings
    # ====================================================
    seed = 42
    seeds = [42, 43, 44]
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    debug = False  # Set to True to run on a small subset for debugging

    # ====================================================
    # Directories & Paths
    # ====================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Working directory for caching intermediate files and saving models
    WORKING_DIR = "./working/idea_14"
    # Directory for final submission
    SUBMISSION_DIR = "./submission"

    # Metadata File Paths
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "validation_metadata.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output Paths
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Create directories if they don't exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ====================================================
    # Model Architecture
    # ====================================================
    model_name = "microsoft/deberta-v3-large"
    # Dropout probability for the classification head
    dropout = 0.1
    # Kernel size for the Gated Convolutional Head
    conv_kernel_size = 3

    # ====================================================
    # Data Processing
    # ====================================================
    # Maximum sequence length for tokenization
    max_len = 128
    # Sigma for Gaussian smoothing of target labels
    smoothing_sigma = 1.0

    # ====================================================
    # Training Hyperparameters
    # ====================================================
    n_folds = 5
    epochs = 4
    train_batch_size = 8
    valid_batch_size = 16

    # Learning Rate & Optimizer
    lr = 5e-5
    weight_decay = 0.01
    eps = 1e-6

    # Scheduler
    scheduler_type = "cosine"
    warmup_ratio = 0.1

    # ====================================================
    # Advanced Optimization Techniques
    # ====================================================
    # Layer-wise Learning Rate Decay factor
    llrd_decay = 0.9
    # Alpha weight for R-Drop (Consistency Regularization) loss
    r_drop_alpha = 1.0
