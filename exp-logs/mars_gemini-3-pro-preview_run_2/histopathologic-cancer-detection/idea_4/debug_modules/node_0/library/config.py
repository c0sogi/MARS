import os
import torch


class Config:
    """
    Configuration class for the Tumor Detection Pipeline.
    Defines hyperparameters, file paths, and system settings.
    """

    # --- General Configuration ---
    seed = 42
    debug = False  # Set to True to run on a small subset for debugging

    # --- Compute ---
    # Utilizing available vCPUs (12) and GPU
    num_workers = 12
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Paths ---
    # Input Directories (Read-Only)
    input_dir = "./input"
    metadata_dir = "./metadata"

    # Metadata Files
    train_metadata_path = os.path.join(metadata_dir, "train.csv")
    val_metadata_path = os.path.join(metadata_dir, "val.csv")
    test_metadata_path = os.path.join(metadata_dir, "test.csv")

    # Output Directories (Writeable)
    # Working directory for this specific idea iteration
    working_dir = "./working/idea_4"
    checkpoint_dir = os.path.join(working_dir, "checkpoints")
    cache_dir = os.path.join(working_dir, "cache")

    # Submission
    submission_dir = "./submission"
    submission_path = os.path.join(submission_dir, "submission.csv")

    # --- Data Parameters ---
    # Image Dimensions
    image_size = 96  # Size to load the image (full patch)
    crop_size = 64  # Size to crop for the model (center crop)

    # Normalization Statistics (Derived from EDA)
    # R, G, B channels
    dataset_mean = [0.7035, 0.5476, 0.6975]
    dataset_std = [0.2388, 0.2821, 0.2159]

    # Augmentation
    mixup_alpha = 0.2

    # --- Model Parameters ---
    model_name = "convnext_tiny"
    pretrained = True
    num_classes = 1
    use_gem_pooling = True  # Use Generalized Mean Pooling instead of GAP

    # --- Training Parameters ---
    n_folds = 5
    epochs = 20
    batch_size = 256

    # Optimization
    learning_rate = 2e-4
    weight_decay = 0.05
    optimizer_name = "AdamW"
    scheduler_name = "CosineAnnealingLR"
    min_lr = 1e-6

    # Regularization
    use_ema = True
    ema_decay = 0.9999

    # --- Inference Parameters ---
    use_tta = True  # Enable Test Time Augmentation
    tta_steps = 8  # 4 rotations * 2 flips

    @classmethod
    def setup(cls):
        """
        Creates necessary output directories if they do not exist.
        """
        os.makedirs(cls.working_dir, exist_ok=True)
        os.makedirs(cls.checkpoint_dir, exist_ok=True)
        os.makedirs(cls.cache_dir, exist_ok=True)
        os.makedirs(cls.submission_dir, exist_ok=True)
