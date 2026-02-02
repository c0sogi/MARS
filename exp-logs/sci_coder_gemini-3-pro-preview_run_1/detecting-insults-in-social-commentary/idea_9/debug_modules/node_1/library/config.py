import os
import torch


class Config:
    """
    Configuration class for the Hybrid DeBERTa-v3 with Structural Fusion
    and Soft-Target Self-Distillation Pipeline (Idea 9).
    """

    # ====================================================
    # General Settings
    # ====================================================
    seed = 42
    debug = False  # Set to True to run on a small subset for debugging
    debug_subset_size = 100
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ====================================================
    # Directories & Paths
    # ====================================================
    # Input Metadata (Read-Only)
    train_meta_path = "./metadata/train.csv"
    val_meta_path = "./metadata/validation.csv"
    test_meta_path = "./metadata/test.csv"

    # Output / Working Directories
    working_dir = "./working/idea_9/"
    output_dir = os.path.join(working_dir, "outputs")
    cache_dir = os.path.join(working_dir, "cache")

    # Model Checkpoint Paths
    teacher_model_dir = os.path.join(output_dir, "teacher")
    student_model_dir = os.path.join(output_dir, "student")

    # Submission
    submission_dir = "./submission"
    submission_path = os.path.join(submission_dir, "submission.csv")

    # ====================================================
    # Model Architecture
    # ====================================================
    model_name = "microsoft/deberta-v3-base"
    max_len = 160  # Sufficient for comment distribution (mean ~196 chars)
    hidden_size = 768

    # Structural Fusion Features
    svd_dim = 256

    # Variable-Rate Multi-Sample Dropout (VR-MSD)
    dropout_rates = [0.1, 0.2, 0.3, 0.4, 0.5]

    # ====================================================
    # Training Hyperparameters
    # ====================================================
    n_folds = 5
    epochs = 4  # Number of training epochs
    batch_size = 16
    gradient_accumulation_steps = 1
    max_grad_norm = 1000
    patience = 3  # Early stopping patience

    # Optimization (Differential Learning Rates)
    lr_backbone = 2e-5
    lr_head = 1e-3
    weight_decay = 0.01
    eps = 1e-6

    # Scheduler
    scheduler_type = "cosine"
    warmup_ratio = 0.1
    num_cycles = 0.5

    # Adversarial Weight Perturbation (AWP)
    use_awp = True
    awp_lr = 1e-4
    awp_eps = 1e-4
    awp_start_epoch = 1

    # ====================================================
    # Distillation Settings
    # ====================================================
    # Soft labels are used, so no hard thresholding parameter is needed here.
    # The student uses a combined dataset of Labeled Train + Soft-Labeled Test.

    @classmethod
    def create_directories(cls):
        """Creates necessary directories for the experiment."""
        os.makedirs(cls.working_dir, exist_ok=True)
        os.makedirs(cls.output_dir, exist_ok=True)
        os.makedirs(cls.cache_dir, exist_ok=True)
        os.makedirs(cls.teacher_model_dir, exist_ok=True)
        os.makedirs(cls.student_model_dir, exist_ok=True)
        os.makedirs(cls.submission_dir, exist_ok=True)

    @classmethod
    def print_config(cls):
        """Prints the current configuration."""
        print("=" * 40)
        print(f"Configuration (Idea 9)")
        print("=" * 40)
        for k, v in cls.__dict__.items():
            if (
                not k.startswith("__")
                and not callable(v)
                and not isinstance(v, (classmethod, staticmethod))
            ):
                print(f"{k:<25}: {v}")
        print("=" * 40)
