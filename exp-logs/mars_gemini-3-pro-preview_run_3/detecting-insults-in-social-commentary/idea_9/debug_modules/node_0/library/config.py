import os
import torch


class ModelConfig:
    """
    Configuration for the Robust Semi-Supervised Heterogeneous Ensemble.
    Centralizes all hyperparameters for data, model, training, and specific strategies
    like AWP and Pseudo-Labeling.
    """

    # ==========================================
    # General Settings
    # ==========================================
    seed = 42
    seeds = [42, 43, 44]  # Seeds for ensemble members
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Debugging
    # Set debug to True to run on a small subset of data for verification
    debug = False
    debug_sample_size = 50

    # ==========================================
    # Directories
    # ==========================================
    # Input metadata containing train/val/test splits
    input_dir = "./metadata"

    # Working directory for this specific idea
    working_dir = "./working/idea_9"

    # Sub-directories for artifacts
    model_dir = os.path.join(working_dir, "models")
    cache_dir = os.path.join(working_dir, "cache")
    submission_dir = os.path.join(working_dir, "submission")

    # Ensure working directories exist
    for _d in [working_dir, model_dir, cache_dir, submission_dir]:
        os.makedirs(_d, exist_ok=True)

    # ==========================================
    # Data Hyperparameters
    # ==========================================
    max_length = 160  # Critical for capturing context in longer comments

    # ==========================================
    # Model Architecture
    # ==========================================
    backbones = ["roberta-large", "microsoft/deberta-v3-large"]

    # Regularization
    dropout = 0.2
    freeze_layers = 6  # Freeze embeddings and bottom 6 encoder layers

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    epochs = 5  # Sufficient epochs for AWP convergence

    # Batch Size Strategy
    # Target Effective Batch Size = 32
    # Adjust train_batch_size based on GPU VRAM (A100 40GB can handle 16 large models)
    train_batch_size = 16
    accumulation_steps = 2

    valid_batch_size = 32

    # Optimization
    learning_rate = 1e-5
    weight_decay = 0.01
    scheduler_type = "linear"
    warmup_ratio = 0.1
    max_grad_norm = 1.0

    # ==========================================
    # Adversarial Weight Perturbation (AWP)
    # ==========================================
    # Enabled in Stage 2
    use_awp = True
    awp_lr = 1e-4
    awp_eps = 1e-2
    awp_start_epoch = (
        1  # Start AWP after the first epoch to allow initial stabilization
    )

    # ==========================================
    # Pseudo-Labeling (Self-Training)
    # ==========================================
    # Thresholds for hard labeling
    pseudo_label_threshold_high = 0.95
    pseudo_label_threshold_low = 0.05

    def __str__(self):
        """Helper to print config state."""
        attributes = [a for a in dir(self) if not a.startswith("__")]
        config_str = "ModelConfig:\n"
        for attr in attributes:
            val = getattr(self, attr)
            if not callable(val):
                config_str += f"  {attr}: {val}\n"
        return config_str
