import os
import torch


class Config:
    """
    Configuration for Apple Disease Detection Task.
    Implements strategy: Heterogeneous Ensemble (EffNetV2-L + ConvNeXt-Base)
    with GeM Pooling, SWA, and Stacking.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    seed = 42
    debug = False  # Set to True to run on a small subset for debugging
    num_folds = 5
    num_workers = 4
    # Use CUDA if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================================================================
    # Directories & Paths
    # =========================================================================
    input_dir = "./input"
    metadata_dir = "./metadata"

    # Working directory for caching and saving models (Idea 9)
    working_dir = "./working/idea_9"
    os.makedirs(working_dir, exist_ok=True)

    # Metadata paths
    train_metadata_path = os.path.join(metadata_dir, "train_metadata.csv")
    val_metadata_path = os.path.join(metadata_dir, "val_metadata.csv")
    test_metadata_path = os.path.join(metadata_dir, "test_metadata.csv")

    # Output paths
    submission_path = "./submission/submission.csv"
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)

    # =========================================================================
    # Data & Targets
    # =========================================================================
    # We decompose the multi-class problem into 2 binary tasks: Rust and Scab.
    # "Multiple Diseases" is inferred when both are high. "Healthy" when both are low.
    target_columns = ["rust", "scab"]
    num_classes = 2

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    batch_size = 16  # Constrained by EffNetV2-L VRAM usage
    epochs = 12  # Sufficient for fine-tuning pre-trained models

    # Mixed Precision & Gradient Accumulation
    use_amp = True
    grad_accumulation_steps = 1

    # Optimizer
    learning_rate = 1e-4
    weight_decay = 1e-4
    min_lr = 1e-6

    # Scheduler
    scheduler_patience = 2
    scheduler_factor = 0.1

    # Early Stopping
    early_stopping_patience = 4

    # Stochastic Weight Averaging (SWA)
    use_swa = True
    swa_start_epoch = 8
    swa_lr = 5e-5

    # =========================================================================
    # Augmentation (Albumentations)
    # =========================================================================
    # CoarseDropout parameters to force distributed feature learning
    coarse_dropout_params = {
        "max_holes": 8,
        "max_height": 100,
        "max_width": 100,
        "min_holes": 1,
        "min_height": 16,
        "min_width": 16,
        "fill_value": 0,
        "p": 0.5,
    }

    # =========================================================================
    # Model Architecture Configurations
    # =========================================================================
    # Heterogeneous Ensemble definitions
    models = [
        {
            # EfficientNetV2-L: Superior compound scaling
            "name": "tf_efficientnetv2_l.in21k_ft_in1k",
            "image_size": 480,  # Native resolution
            "use_gem": True,  # Generalized Mean Pooling
            "dropout_rate": 0.3,
            "drop_path_rate": 0.2,
        },
        {
            # ConvNeXt-Base: Large kernel inductive bias
            "name": "convnext_base.fb_in22k_ft_in1k_384",
            "image_size": 384,  # Native resolution
            "use_gem": True,  # Generalized Mean Pooling
            "dropout_rate": 0.3,
            "drop_path_rate": 0.2,
        },
    ]

    # =========================================================================
    # Stacking Meta-Learner
    # =========================================================================
    # Logistic Regression parameters for calibrating OOF predictions
    meta_learner_params = {
        "C": 1.0,
        "penalty": "l2",
        "solver": "lbfgs",
        "max_iter": 1000,
        "random_state": seed,
        "class_weight": "balanced",
    }
