import os
import torch


class Config:
    # =========================================================================
    # General Settings
    # =========================================================================
    seed = 42
    debug = False  # Set to True to run on a small subset for debugging
    debug_subset_size = 100
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================================================================
    # Paths
    # =========================================================================
    # Input directories (Read-only)
    input_dir = "./metadata"
    train_path = os.path.join(input_dir, "train.csv")
    val_path = os.path.join(input_dir, "val.csv")
    test_path = os.path.join(input_dir, "test.csv")

    # Output directory (Write allowed)
    output_dir = "./working/idea_6"
    submission_path = "./submission/submission.csv"

    # =========================================================================
    # Model Architecture
    # =========================================================================
    model_name = "microsoft/deberta-v3-large"
    max_len = 128
    target_col = "Insult"
    text_col = "Comment"
    num_classes = 1

    # Classification Head
    pooler_type = "mean"  # Options: "mean", "cls", "max"
    use_msd = True  # Multi-Sample Dropout
    msd_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
    fc_dropout = 0.0  # Standard dropout if MSD is False

    # =========================================================================
    # Stage 1: Task-Adaptive Pre-Training (TAPT)
    # =========================================================================
    use_tapt = True
    tapt_epochs = 5
    tapt_batch_size = 8  # Adjusted for Large model on 40GB GPU
    tapt_lr = 2e-5
    tapt_weight_decay = 0.01
    mlm_probability = 0.15
    tapt_output_dir = os.path.join(output_dir, "tapt_model")

    # =========================================================================
    # Stage 2: Supervised Fine-Tuning (SFT)
    # =========================================================================
    n_folds = 5
    epochs = 6
    train_batch_size = 8  # Adjusted for Large model + AWP overhead
    valid_batch_size = 16

    # Optimizer (AdamW)
    learning_rate = 1e-5
    weight_decay = 0.01
    eps = 1e-6
    betas = (0.9, 0.999)
    max_grad_norm = 1.0

    # Scheduler
    scheduler_type = "cosine"
    warmup_ratio = 0.1

    # Layer-wise Learning Rate Decay (LLRD)
    use_llrd = True
    llrd_decay = 0.9

    # Adversarial Weight Perturbation (AWP)
    use_awp = True
    awp_start_epoch = 2  # Start AWP after epoch 1 (1-based counting in logs, so > 1.0)
    awp_lr = 1e-4
    awp_eps = 1e-2

    # =========================================================================
    # Methods
    # =========================================================================
    @classmethod
    def setup(cls):
        """Ensure output directories exist."""
        os.makedirs(cls.output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(cls.submission_path), exist_ok=True)
        if cls.use_tapt:
            os.makedirs(cls.tapt_output_dir, exist_ok=True)
