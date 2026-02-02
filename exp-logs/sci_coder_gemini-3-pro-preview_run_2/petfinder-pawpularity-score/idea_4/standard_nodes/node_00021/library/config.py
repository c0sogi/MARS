import os
import torch


class Config:
    """
    Central configuration for the Pet Pawpularity Prediction task.
    Handles hyperparameters, file paths, and model settings for the
    Tri-Model Stacking strategy.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    seed = 42
    debug = False  # Set to True to run on a small subset of data for debugging
    debug_sample_size = 100
    num_workers = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================================================================
    # Data Paths
    # =========================================================================
    # Input directories (Read-Only)
    input_dir = "./input"

    # Metadata paths (Pre-generated)
    train_metadata_path = "./metadata/train.csv"
    val_metadata_path = "./metadata/validation.csv"
    test_metadata_path = "./metadata/test.csv"

    # Output directories (Writeable)
    output_dir = "./working/idea_4"
    submission_dir = "./submission"
    submission_path = os.path.join(submission_dir, "submission.csv")

    # =========================================================================
    # Data Configuration
    # =========================================================================
    image_size = 384
    target_col = "Pawpularity"

    # =========================================================================
    # Model Architectures (timm names)
    # =========================================================================
    # 1. Swin Transformer Large: Hierarchical attention
    # 2. ConvNeXt Large: Pure CNN inductive bias
    # 3. BEiT Large: Masked Image Modeling pre-training
    models = {
        "swin": "swin_large_patch4_window12_384",
        "convnext": "convnext_large_384_in22ft1k",
        "beit": "beit_large_patch16_384",
    }

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    n_folds = 5
    epochs = 10
    batch_size = 16  # Adjusted for Large models @ 384x384 on A100-40GB

    # Optimization
    # Differential learning rates for backbone and head
    backbone_lr = 1e-5
    head_lr = 1e-4
    min_lr = 1e-7
    weight_decay = 1e-6

    # Scheduler
    T_max = epochs  # For CosineAnnealingLR

    # Early Stopping
    patience = 3
    verbose = True

    # =========================================================================
    # Meta-Learner Hyperparameters
    # =========================================================================
    meta_alpha = 1.0  # Regularization strength for Ridge Regression

    def __init__(self):
        """
        Initialize configuration and ensure necessary writeable directories exist.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.submission_dir, exist_ok=True)

    def get_model_path(self, model_name, fold):
        """
        Helper to generate standardized model save paths.
        """
        filename = f"{model_name}_fold_{fold}.pth"
        return os.path.join(self.output_dir, filename)

    def get_oof_path(self, model_name):
        """
        Helper to generate path for Out-Of-Fold predictions.
        """
        return os.path.join(self.output_dir, f"oof_{model_name}.csv")
