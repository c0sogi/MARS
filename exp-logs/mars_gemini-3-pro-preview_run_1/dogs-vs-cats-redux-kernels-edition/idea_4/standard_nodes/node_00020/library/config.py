import os
import torch


class CFG:
    """
    Configuration class for the Dog vs. Cat classification task.
    Centralizes hyperparameters, file paths, and model settings.
    """

    # ====================================================
    # General Settings
    # ====================================================
    seed = 42
    debug = False
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ====================================================
    # Data & Paths
    # ====================================================
    input_dir = "./input"
    metadata_dir = "./metadata"

    # Metadata files
    train_csv = os.path.join(metadata_dir, "train.csv")
    val_csv = os.path.join(
        metadata_dir, "val.csv"
    )  # Explicit validation set if needed, though we use CV
    test_csv = os.path.join(metadata_dir, "test.csv")

    # Output directories
    # Using 'idea_4' as the current experiment version
    working_dir = "./working/idea_4"
    output_dir = os.path.join(working_dir, "models")
    submission_dir = "./submission"

    # Ensure directories exist
    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(submission_dir, exist_ok=True)

    # ====================================================
    # Model Architecture
    # ====================================================
    # Tri-Architecture Heterogeneous Ensemble
    model_names = [
        "tf_efficientnetv2_m.in21k_ft_in1k",
        "convnext_base.fb_in1k",
        "swinv2_base_window12to24_192to384.ms_in22k_ft_in1k",
    ]

    # ====================================================
    # Training Hyperparameters
    # ====================================================
    image_size = 384
    batch_size = 32
    epochs = 3
    n_folds = 5

    # Optimization
    lr = 1e-4
    min_lr = 1e-6
    weight_decay = 1e-6

    # Scheduler
    scheduler_type = "CosineAnnealingLR"
    T_max = epochs  # For CosineAnnealingLR

    # ====================================================
    # Inference / Ensemble
    # ====================================================
    use_tta = True  # Test Time Augmentation (Horizontal Flip)

    # ====================================================
    # Caching
    # ====================================================
    # Path for caching processed datasets if needed
    cache_dir = working_dir
