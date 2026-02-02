import os
import torch
import random
import numpy as np


class CFG:
    # ====================================================
    # General Configuration
    # ====================================================
    idea_name = "idea_10"
    seed = 42
    debug = False  # Set to True for fast debugging on a subset
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_freq = 50

    # ====================================================
    # Data Paths & Caching
    # ====================================================
    # We use the generated metadata CSVs which contain the splits and text
    metadata_dir = "./metadata"
    train_path = os.path.join(metadata_dir, "train.csv")
    val_path = os.path.join(metadata_dir, "val.csv")
    test_path = os.path.join(metadata_dir, "test.csv")
    sample_submission_path = "./input/sample_submission.csv"

    # Output directory for saving models, logs, and cached data
    output_dir = os.path.join("./working", idea_name)
    os.makedirs(output_dir, exist_ok=True)

    # Cache directory for processed features (parquet/npy)
    cache_dir = os.path.join(output_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    # ====================================================
    # Model Architecture
    # ====================================================
    model_name = "microsoft/deberta-v3-large"
    max_len = 320  # Sufficient for comment classification, balances memory
    dropout = 0.1

    # Column Definitions
    target_col = "target"
    binary_target_col = "binary_target"

    # Identity attributes used for Bias AUC metrics and Auxiliary Heads
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

    # Auxiliary attribute for Multi-Task Learning
    aux_attack_col = "identity_attack"

    # ====================================================
    # Training Dynamics (Curriculum Pipeline)
    # ====================================================
    # Global Training Settings
    train_batch_size = 8  # Conservative for A100 40GB + Large Model
    valid_batch_size = 16
    gradient_accumulation_steps = 1
    max_grad_norm = 1.0
    weight_decay = 0.01

    # --- Stage 1: Domain-Adaptive Pretraining (DAPT) ---
    # Unsupervised MLM on Train + Test text
    dapt_epochs = 3
    dapt_lr = 2e-5
    mlm_probability = 0.15

    # --- Stage 2: General Multi-Task Fine-Tuning ---
    # Standard training with Uniform Stratified Sampling
    stage2_epochs = 2
    stage2_lr = 1e-5
    encoder_lr = 1e-5
    decoder_lr = 5e-5
    min_lr = 1e-7
    scheduler = "cosine"
    warmup_ratio = 0.1

    # --- Stage 3: Robust Metric Optimization ---
    # Training with Bias-Focused Sampling, Ranking Loss, and AWP
    stage3_epochs = 2
    stage3_lr = 5e-6  # Lower LR for final refinement

    # ====================================================
    # Loss Weights & Objectives
    # ====================================================
    # Multi-Task Learning Weights
    toxicity_loss_weight = 1.0
    aux_identity_loss_weight = 0.2
    aux_attack_loss_weight = 0.2

    # Robust Optimization Weights (Stage 3)
    ranking_loss_weight = 0.5  # Weight for Margin Ranking Loss

    # Sample Weighting
    # Multiplier for "Bias Trap" examples (Toxic+Identity or NonToxic+Identity)
    bias_sample_weight = 5.0

    # ====================================================
    # Advanced Regularization
    # ====================================================
    # Exponential Moving Average (EMA)
    use_ema = True
    ema_decay = 0.999

    # Adversarial Weight Perturbation (AWP)
    # Applied primarily in Stage 3 to flatten the loss landscape
    use_awp = True
    awp_lr = 1e-4
    awp_eps = 1e-2
    awp_adv_step = 1

    # ====================================================
    # Utilities
    # ====================================================
    @staticmethod
    def set_seed(seed=42):
        """Sets the seed for reproducibility across all libraries."""
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
