import os
import torch


class Config:
    """
    Configuration class for the Multi-Task Deberta-V3-Large model pipeline.
    Centralizes all hyperparameters for data, model, training, and advanced regularization techniques.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    seed = 42
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    debug = False  # Set to True to run on a small subset of data for debugging

    # =========================================================================
    # Data Paths
    # =========================================================================
    # Using the pre-generated metadata files
    train_path = "./metadata/train.csv"
    val_path = "./metadata/val.csv"
    test_path = "./metadata/test.csv"
    sample_submission_path = "./input/sample_submission.csv"

    # Output directory for caching and model artifacts
    output_dir = "./working/idea_4"

    # =========================================================================
    # Model Architecture
    # =========================================================================
    model_name = "microsoft/deberta-v3-large"
    gradient_checkpointing = False

    # Input dimensions
    # Context + Anchor + Target.
    # Max length analysis showed mean ~16 chars, max ~100 chars.
    # 140 tokens is sufficient for [CLS] + Context + [SEP] + Anchor + [SEP] + Target + [SEP]
    max_len = 140

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    epochs = 5
    train_batch_size = 8  # Adjusted for A100 40GB with Deberta-Large
    valid_batch_size = 16

    # Optimization
    learning_rate = 2e-5
    weight_decay = 0.01
    eps = 1e-6
    betas = (0.9, 0.999)
    max_grad_norm = 1000

    # Scheduler
    scheduler_type = "cosine"
    warmup_ratio = 0.1
    num_cycles = 0.5

    # =========================================================================
    # Advanced Techniques
    # =========================================================================

    # Layer-wise Learning Rate Decay (LLRD)
    # Decays LR for lower layers to preserve pre-trained knowledge
    use_llrd = True
    llrd_decay = 0.9

    # Multi-Sample Dropout (MSD)
    # Accelerates convergence and acts as an ensemble within the model
    use_msd = True
    num_msd_rounds = 5
    msd_dropout_rate = 0.1
    fc_dropout = 0.0  # Standard dropout before MSD (often 0 if using MSD)

    # Adversarial Weight Perturbation (AWP)
    # Perturbs weights to find a flatter, more robust loss minimum
    use_awp = True
    awp_lr = 1e-4
    awp_eps = 1e-4
    awp_start_epoch = 1.0  # Start AWP after the 1st epoch

    # =========================================================================
    # Multi-Task Objectives
    # =========================================================================
    # We predict both the continuous score (Regression) and the discrete bucket (Classification)
    num_classes = 5  # Classes: 0.0, 0.25, 0.50, 0.75, 1.0

    # Loss Weights
    # Total Loss = (mse_weight * MSE) + (ce_weight * CrossEntropy)
    mse_weight = 1.0
    ce_weight = 0.5

    @classmethod
    def create_output_dir(cls):
        """Ensures the output directory exists."""
        os.makedirs(cls.output_dir, exist_ok=True)
