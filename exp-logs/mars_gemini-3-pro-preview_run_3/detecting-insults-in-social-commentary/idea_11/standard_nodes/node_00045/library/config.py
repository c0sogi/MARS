import os
import torch


class Config:
    """
    Configuration class for the Corrected Adversarial Knowledge Distillation Ensemble.
    Centralizes all hyperparameters for Model, Data, Training, and Evaluation.
    """

    # ==========================================
    # General Settings
    # ==========================================
    project_name = "insult_detection_idea_11"
    seed = 42
    seeds = [42, 43, 44]  # Seeds for the ensemble members
    debug = False  # Set to True for quick debugging with small data
    debug_sample_size = 50

    # Hardware
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==========================================
    # Data Paths
    # ==========================================
    # Input metadata paths (read-only)
    train_path = "./metadata/train.csv"
    val_path = "./metadata/val.csv"
    test_path = "./metadata/test.csv"

    # Working directory for this specific idea
    working_dir = "./working/idea_11/"

    # Sub-directories for artifacts
    output_dir = os.path.join(working_dir, "models")
    cache_dir = os.path.join(working_dir, "cache")

    # Final submission file
    submission_path = os.path.join(working_dir, "submission.csv")

    # ==========================================
    # Model Architecture
    # ==========================================
    # Heterogeneous backbones
    model_names = ["roberta-large", "microsoft/deberta-v3-large"]

    # Architecture specifics
    max_len = 160  # Critical for capturing context
    dropout = 0.2  # Fixed dropout for regularization
    pooling = "mean"  # Mean pooling of final hidden states
    freeze_layers = 6  # Freeze embeddings and bottom 6 layers
    num_classes = 1  # Binary classification

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    # Effective Batch Size = train_batch_size * gradient_accumulation_steps
    # 8 * 4 = 32 (Target Effective Batch Size)
    train_batch_size = 8
    valid_batch_size = 16
    gradient_accumulation_steps = 4

    # Optimization
    learning_rate = 1e-5  # Learning rate for backbones
    weight_decay = 0.01
    max_grad_norm = 1.0

    # Scheduler: Linear Decay (Warmup + Decay to 0)
    scheduler_type = "linear"
    warmup_ratio = 0.1

    # ==========================================
    # Stage 1: Teacher Ensemble Training
    # ==========================================
    epochs_stage1 = 5  # Train teachers for 5 epochs

    # ==========================================
    # Stage 2: Adversarial Student Distillation
    # ==========================================
    epochs_stage2 = 5  # Train students for 5 epochs (corrected from 2)

    # Distillation
    distillation_alpha = 0.5  # Weight for Soft Target Loss vs Hard Label Loss
    temperature = 1.0  # Temperature for soft targets

    # Adversarial Weight Perturbation (AWP)
    use_awp = True
    awp_lr = 1e-4
    awp_eps = 1e-2
    awp_start_epoch = 1  # Start AWP after 1 epoch of stabilization

    def __init__(self):
        """
        Initialize configuration and ensure necessary directories exist.
        """
        os.makedirs(self.working_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
