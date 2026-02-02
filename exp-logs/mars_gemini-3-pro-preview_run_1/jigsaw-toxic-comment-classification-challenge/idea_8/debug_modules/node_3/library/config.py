import os
import torch


class Config:
    # =========================================================================
    # General System Settings
    # =========================================================================
    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 4

    # Debugging
    debug = False  # Set to True to run on a small subset for testing
    debug_sample_size = 1000

    # =========================================================================
    # File Paths & Directories
    # =========================================================================
    # Input directories (Read-Only)
    input_dir = "./input"
    metadata_dir = "./metadata"

    # Metadata files
    train_meta_path = os.path.join(metadata_dir, "train.csv")
    val_meta_path = os.path.join(metadata_dir, "val.csv")
    test_meta_path = os.path.join(metadata_dir, "test.csv")

    # Raw data files
    train_raw_path = os.path.join(input_dir, "train.csv")
    test_raw_path = os.path.join(input_dir, "test.csv")
    sample_submission_path = os.path.join(input_dir, "sample_submission.csv")

    # Working directory for caching and outputs (Write Access)
    working_dir = "./working/idea_8"
    output_dir = "./working/idea_8/outputs"

    # Ensure working directories exist
    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # =========================================================================
    # Data Configuration
    # =========================================================================
    target_cols = [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ]
    num_classes = len(target_cols)

    # Text processing
    max_len = 512  # Max sequence length for DeBERTa

    # =========================================================================
    # Model Architecture
    # =========================================================================
    model_name = "microsoft/deberta-v3-base"
    hidden_dropout_prob = 0.1
    attention_probs_dropout_prob = 0.1

    # Layer Aggregation
    num_layers_to_aggregate = 4  # Aggregate the last N layers

    # Multi-Sample Dropout
    msd_num = 5  # Number of dropout masks

    # =========================================================================
    # Training Hyperparameters (Supervised)
    # =========================================================================
    n_folds = 5
    epochs = 5
    batch_size = 16
    accumulate_grad_batches = 1

    # Optimizer (AdamW)
    lr = 2e-5
    weight_decay = 0.01
    eps = 1e-6
    betas = (0.9, 0.999)

    # Scheduler (OneCycleLR)
    pct_start = 0.1  # Percentage of training to increase LR
    div_factor = 25.0
    final_div_factor = 1000.0

    # Gradient Clipping
    max_grad_norm = 1.0

    # =========================================================================
    # Adversarial Weight Perturbation (AWP)
    # =========================================================================
    use_awp = True
    awp_start_epoch = 2  # Start AWP after this many epochs
    awp_lr = 1e-4  # Learning rate for the adversary
    awp_eps = 1e-4  # Epsilon for AWP

    # =========================================================================
    # Domain-Adaptive Pre-training (MLM)
    # =========================================================================
    mlm_epochs = 3
    mlm_batch_size = 16
    mlm_lr = 2e-5
    mlm_probability = 0.15  # Masking probability

    # Path to save/load the domain-adapted backbone
    mlm_model_dir = os.path.join(working_dir, "mlm_backbone")
    os.makedirs(mlm_model_dir, exist_ok=True)
