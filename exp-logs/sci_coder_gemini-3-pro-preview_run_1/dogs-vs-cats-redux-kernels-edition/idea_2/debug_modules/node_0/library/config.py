import os
import torch


class Config:
    """
    Configuration class for the Dog vs Cat classification pipeline.
    Centralizes all hyperparameters, file paths, and system settings.
    """

    # -------------------------------------------------------------------------
    # General Setup
    # -------------------------------------------------------------------------
    seed = 42
    debug = False  # Set to True to use a small subset of data for debugging/testing

    # -------------------------------------------------------------------------
    # Directory Paths
    # -------------------------------------------------------------------------
    input_dir = "./input"
    metadata_dir = "./metadata"
    working_dir = "./working/idea_2"
    submission_dir = "./submission"

    # Ensure necessary output directories exist
    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(submission_dir, exist_ok=True)

    # Directory for saving model checkpoints
    model_dir = os.path.join(working_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    # Paths to metadata files
    train_metadata_path = os.path.join(metadata_dir, "train.csv")
    val_metadata_path = os.path.join(metadata_dir, "val.csv")
    test_metadata_path = os.path.join(metadata_dir, "test.csv")

    # -------------------------------------------------------------------------
    # Model Architecture
    # -------------------------------------------------------------------------
    backbone = "tf_efficientnetv2_m"  # Medium variant for higher capacity
    pretrained = True
    image_size = 384  # Increased resolution for better detail capture
    num_classes = 1  # Binary classification (Dog vs Cat)

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    n_folds = 5  # 5-Fold Cross-Validation
    epochs = 4  # Short training duration per fold
    batch_size = 32  # Optimized for A100 40GB VRAM with 384x384 images

    # Optimizer settings (AdamW)
    learning_rate = 1e-4
    weight_decay = 1e-2

    # Scheduler settings (Cosine Annealing)
    min_lr = 1e-6
    T_max = epochs  # Cycle length matches total epochs

    # -------------------------------------------------------------------------
    # Hardware & System
    # -------------------------------------------------------------------------
    num_workers = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------------------------------------------------------
    # Inference Strategy
    # -------------------------------------------------------------------------
    use_tta = True  # Enable Test Time Augmentation (Horizontal Flip)
