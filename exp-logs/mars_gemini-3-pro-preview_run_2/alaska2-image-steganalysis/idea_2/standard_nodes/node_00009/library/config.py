import os
import torch


class Config:
    """
    Configuration for Steganography Detection Task.
    Centralizes all hyperparameters, file paths, and model settings.
    """

    # -------------------------------------------------------------------------
    # General Settings
    # -------------------------------------------------------------------------
    seed = 42
    debug = False  # Set to True to run with a smaller subset for debugging

    # -------------------------------------------------------------------------
    # Compute Resources
    # -------------------------------------------------------------------------
    num_workers = 12
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------------------
    # File Paths & Directories
    # -------------------------------------------------------------------------
    input_root = "./input"
    metadata_dir = "./metadata"

    # Metadata CSVs
    train_csv = os.path.join(metadata_dir, "train.csv")
    val_csv = os.path.join(metadata_dir, "val.csv")
    test_csv = os.path.join(metadata_dir, "test.csv")

    # Working Directories
    # Using 'idea_2' to align with the caching/experiment iteration
    working_dir = "./working/idea_2"
    checkpoint_dir = os.path.join(working_dir, "checkpoints")
    predictions_dir = os.path.join(working_dir, "predictions")

    # Submission Output
    submission_path = "./submission/submission.csv"

    # -------------------------------------------------------------------------
    # Data Configuration
    # -------------------------------------------------------------------------
    image_size = 512

    # Dataset Strategy
    # "dynamic_pairing": For every epoch, pair Cover with 1 random Stego variant
    training_strategy = "dynamic_pairing"

    # Debugging Parameters
    # If debug is True, these define the subset size
    debug_train_size = 2000
    debug_val_size = 500

    # -------------------------------------------------------------------------
    # Model Architecture
    # -------------------------------------------------------------------------
    # Backbone: EfficientNetV2-Small (TensorFlow weights port)
    backbone_name = "tf_efficientnetv2_s"
    pretrained = True

    # Custom Head / Stem
    use_srm_stem = True  # Use fixed SRM filters at input
    use_gem_pooling = True  # Use Generalized Mean Pooling

    num_classes = 1  # Binary Classification (Cover vs Stego)

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    epochs = 40

    # Batch Size
    # Reduced to 16 (effective 32) to avoid OOM on A100 with 512x512
    train_batch_size = 16
    val_batch_size = 32

    # Optimizer: AdamW
    optimizer_name = "AdamW"
    lr = 1e-3
    weight_decay = 1e-2

    # Scheduler: Cosine Annealing
    scheduler_name = "CosineAnnealingLR"
    min_lr = 1e-6
    T_max = epochs  # Scheduler cycle matches total epochs

    # -------------------------------------------------------------------------
    # Metrics & Evaluation
    # -------------------------------------------------------------------------
    # Weighted AUC Parameters
    # Thresholds for TPR regions
    tpr_thresholds = [0.0, 0.4, 1.0]
    # Weights for each region (0-0.4 gets 2x weight)
    auc_weights = [2.0, 1.0]

    # Inference Strategy
    # 5-View TTA: Original, H-Flip, V-Flip, Rot90, Rot270
    tta_views = 5

    @classmethod
    def setup(cls):
        """
        Initializes the environment:
        1. Creates necessary directories.
        2. Sets random seeds for reproducibility.
        """
        # Create directories
        os.makedirs(cls.working_dir, exist_ok=True)
        os.makedirs(cls.checkpoint_dir, exist_ok=True)
        os.makedirs(cls.predictions_dir, exist_ok=True)
        os.makedirs(os.path.dirname(cls.submission_path), exist_ok=True)

        # Set seeds
        import random
        import numpy as np

        random.seed(cls.seed)
        np.random.seed(cls.seed)
        torch.manual_seed(cls.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cls.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        print(f"Configuration initialized. Working directory: {cls.working_dir}")
