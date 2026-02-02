import os
import torch


class Config:
    # =========================================================================
    # General Settings
    # =========================================================================
    seed = 42
    debug = False  # Set to True for fast debugging on a subset
    debug_sample_size = 2000  # Number of samples to use in debug mode
    num_workers = 8  # Number of CPU workers for data loading
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================================================================
    # Paths
    # =========================================================================
    input_dir = "./input"
    metadata_dir = "./metadata"

    # Metadata files
    train_csv = os.path.join(metadata_dir, "train.csv")
    val_csv = os.path.join(metadata_dir, "val.csv")
    test_csv = os.path.join(metadata_dir, "test.csv")

    # Image directories
    cover_dir = os.path.join(input_dir, "Cover")
    jmipod_dir = os.path.join(input_dir, "JMiPOD")
    juniward_dir = os.path.join(input_dir, "JUNIWARD")
    uerd_dir = os.path.join(input_dir, "UERD")
    test_dir = os.path.join(input_dir, "Test")

    # Output directories
    working_dir = "./working/idea_3"
    checkpoint_dir = os.path.join(working_dir, "checkpoints")
    predictions_dir = os.path.join(working_dir, "predictions")
    cache_dir = os.path.join(working_dir, "cache")
    submission_path = "./submission/submission.csv"

    # Create directories if they don't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)

    # =========================================================================
    # Model Configuration
    # =========================================================================
    model_name = "efficientnetv2_rw_s"
    target_size = 1  # Binary classification (Cover vs Stego)
    pretrained = True
    drop_rate = 0.3
    drop_path_rate = 0.2

    # GeM Pooling
    use_gem = True
    gem_p_init = 3.0
    gem_trainable = True

    # =========================================================================
    # Data Configuration
    # =========================================================================
    image_size = 512
    input_channels = 3

    # Unique Content Sampling: If True, ensures only one variant of an image ID
    # is seen per epoch (randomly selected from Cover/JMiPOD/JUNIWARD/UERD)
    unique_content_sampling = True

    # =========================================================================
    # Training Configuration
    # =========================================================================
    epochs = 30
    batch_size = 24  # Tuned for A100 40GB with 512x512 images

    # Optimizer
    optimizer_name = "AdamW"
    learning_rate = 2e-4
    weight_decay = 1e-2
    max_grad_norm = 10.0

    # Scheduler
    scheduler_name = "CosineAnnealingLR"
    min_lr = 1e-6
    T_max = epochs  # Cycle length matches epochs

    # =========================================================================
    # Metric Configuration (Weighted AUC)
    # =========================================================================
    # Weights for regions: [0.0-0.4] gets 2x, [0.4-1.0] gets 1x
    tpr_thresholds = [0.0, 0.4, 1.0]
    metric_weights = [2, 1]

    # =========================================================================
    # Inference Configuration
    # =========================================================================
    tta_views = 5  # Original, HFlip, VFlip, Rot90, Rot270
