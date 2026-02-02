import os
import torch


class Config:
    # ====================================================
    # General Settings
    # ====================================================
    seed = 42
    debug = False  # Set to True to run on a small subset for debugging
    debug_sample_size = 50  # Number of samples to use in debug mode
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ====================================================
    # Directory Paths
    # ====================================================
    # Input Metadata (Read-Only)
    metadata_dir = "./metadata"
    train_path = os.path.join(metadata_dir, "train.csv")
    val_path = os.path.join(metadata_dir, "validation.csv")
    test_path = os.path.join(metadata_dir, "test.csv")

    # Working Directory (Artifacts)
    working_dir = "./working/idea_11"
    cache_dir = os.path.join(working_dir, "cache")
    model_dir = os.path.join(working_dir, "models")
    predictions_dir = os.path.join(working_dir, "predictions")

    # Submission Directory
    submission_dir = "./submission"
    submission_path = os.path.join(submission_dir, "submission.csv")

    # Ensure directories exist
    for path in [working_dir, cache_dir, model_dir, predictions_dir, submission_dir]:
        os.makedirs(path, exist_ok=True)

    # ====================================================
    # Data Processing & Feature Engineering
    # ====================================================
    max_len = 128  # Max token length for comments

    # SVD Feature Parameters
    svd_components = 256
    ngram_range_word = (1, 2)
    ngram_range_char = (3, 5)

    # ====================================================
    # Model Architecture
    # ====================================================
    # Backbone A: DeBERTa-v3-Large (SentencePiece tokenizer)
    model_a_name = "microsoft/deberta-v3-large"

    # Backbone B: RoBERTa-Large (Byte-Level BPE tokenizer)
    model_b_name = "roberta-large"

    # ====================================================
    # Training Hyperparameters
    # ====================================================
    n_folds = 5
    epochs = 4

    # Batch Sizes (Adjusted for A100 40GB)
    train_batch_size = 8
    valid_batch_size = 16
    grad_accumulation_steps = 2  # Effective batch size = 16

    max_grad_norm = 1.0

    # Differential Learning Rates
    lr_backbone = 1e-5
    lr_head = 1e-3
    weight_decay = 0.01

    # Scheduler
    warmup_ratio = 0.1

    # Adversarial Weight Perturbation (AWP)
    use_awp = True
    awp_lr = 1e-4
    awp_eps = 1e-2
    awp_start_epoch = 1  # Start AWP after the first epoch

    # Variable-Rate Multi-Sample Dropout (VR-MSD)
    dropout_rates = [0.1, 0.2, 0.3, 0.4, 0.5]

    # ====================================================
    # Stacking / Meta-Learner
    # ====================================================
    meta_alpha = 1.0  # L2 Regularization strength for Ridge Regression
