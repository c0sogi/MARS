import os
import torch


class Config:
    """
    Configuration class for the Toxicity Classification with Bias Mitigation task.
    Centralizes all hyperparameters, file paths, and model settings.
    """

    # ==========================================
    # General Settings
    # ==========================================
    seed = 42
    debug = False  # Set to True to run on a small subset for debugging
    debug_subset_size = 5000  # Number of samples to use when debug=True
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==========================================
    # File Paths
    # ==========================================
    # Input directories (Read-Only)
    input_dir = "./input"
    metadata_dir = "./metadata"

    # Data Paths (using generated metadata)
    train_path = os.path.join(metadata_dir, "train.csv")
    val_path = os.path.join(metadata_dir, "val.csv")
    test_path = os.path.join(metadata_dir, "test.csv")

    # Output directories
    working_dir = "./working/idea_11"
    submission_dir = "./submission"

    # Ensure output directories exist
    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(submission_dir, exist_ok=True)

    # Final Submission Path
    submission_path = os.path.join(submission_dir, "submission.csv")

    # Cache Paths (for deterministic processing)
    # Using parquet for efficient storage of processed dataframes
    cached_train_path = os.path.join(working_dir, "train_processed.parquet")
    cached_val_path = os.path.join(working_dir, "val_processed.parquet")
    cached_test_path = os.path.join(working_dir, "test_processed.parquet")

    # ==========================================
    # Model Architecture
    # ==========================================
    model_name = "microsoft/deberta-v3-large"
    tokenizer_name = "microsoft/deberta-v3-large"

    # Scout model for mining hard negatives (lighter model for speed)
    scout_model_name = "microsoft/deberta-v3-base"

    # Input sequence length
    # EDA shows 95th percentile char length is ~952.
    # 320 tokens is sufficient to cover >95% of samples without truncation.
    max_len = 320

    # ==========================================
    # Column Definitions
    # ==========================================
    target_col = "target"
    binary_target_col = "binary_target"
    text_col = "comment_text"

    # Identity attributes for Bias AUC calculation and auxiliary heads
    identity_cols = [
        "male",
        "female",
        "homosexual_gay_or_lesbian",
        "christian",
        "jewish",
        "muslim",
        "black",
        "white",
        "psychiatric_or_mental_illness",
    ]

    # Auxiliary toxicity subtypes
    aux_cols = [
        "severe_toxicity",
        "obscene",
        "threat",
        "insult",
        "identity_attack",
        "sexual_explicit",
    ]

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    # Batch Sizes (Adjusted for A100 40GB)
    train_batch_size = 8
    valid_batch_size = 16
    accumulate_grad_batches = 1

    # Optimization
    lr = 1e-5
    min_lr = 1e-7
    weight_decay = 0.01
    max_grad_norm = 1.0

    # Scheduler
    scheduler_type = "cosine"
    warmup_ratio = 0.05

    # ==========================================
    # Pipeline Stages Config
    # ==========================================

    # Stage 1: Domain-Adaptive Pretraining (DAPT)
    # Masked Language Modeling on combined Train + Test text
    dapt_epochs = 3
    dapt_lr = 2e-5
    dapt_mask_probability = 0.15

    # Stage 2: Scout Training & Mining
    # Training a lighter model to identify "Bias Traps" in unlabeled data
    scout_epochs = 1
    scout_lr = 2e-5
    mining_threshold_identity = 0.5  # Prob > 0.5 implies identity mention
    mining_threshold_toxicity = 0.1  # Prob < 0.1 implies non-toxic

    # Stage 3: Final Robust Training
    epochs = 4

    # Loss Function Weights
    # Total Loss = BCE + lambda_rank * RankLoss + lambda_aux * AuxLoss
    lambda_rank = 0.5  # Weight for Pairwise Ranking Loss
    lambda_aux = 0.2  # Weight for Auxiliary Identity/Subtype Heads

    # Adversarial Weight Perturbation (AWP)
    use_awp = True
    awp_start_epoch = 1  # Start AWP after the first epoch
    awp_eps = 1e-4  # Epsilon for weight perturbation
    awp_lr = 1e-4  # Learning rate for the adversary
