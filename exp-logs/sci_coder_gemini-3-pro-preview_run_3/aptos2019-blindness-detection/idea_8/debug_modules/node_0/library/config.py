import os
import torch


class Config:
    """
    Configuration class for the Progressive High-Resolution GeM-ConvNeXt pipeline.
    """

    # ==========================================
    # General Configuration
    # ==========================================
    seed = 42
    debug = False  # Set to True to run on a small subset for debugging
    n_folds = 5
    num_workers = 4  # Based on 12 vCPUs available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==========================================
    # Directory Paths
    # ==========================================
    # Input directories (Read-Only)
    input_dir = "./input"
    train_images_dir = os.path.join(input_dir, "train_images")
    test_images_dir = os.path.join(input_dir, "test_images")

    # Metadata paths (Pre-generated)
    metadata_dir = "./metadata"
    train_metadata_path = os.path.join(metadata_dir, "train.csv")
    val_metadata_path = os.path.join(metadata_dir, "val.csv")
    test_metadata_path = os.path.join(metadata_dir, "test.csv")
    sample_submission_path = os.path.join(input_dir, "sample_submission.csv")

    # Working directory (Write allowed)
    # Specific directory for this idea iteration
    working_dir = "./working/idea_8"
    models_dir = os.path.join(working_dir, "models")
    predictions_dir = os.path.join(working_dir, "predictions")
    cache_dir = os.path.join(working_dir, "cache")

    # Submission output
    submission_path = "./submission.csv"

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    model_name = "convnext_base"  # Backbone architecture
    pretrained = True
    num_classes = 1  # Regression output
    drop_rate = 0.0
    drop_path_rate = 0.1  # Stochastic depth rate

    # ==========================================
    # Training Hyperparameters - Stage 1 (Structure Learning)
    # ==========================================
    # Initial training on lower resolution to learn structural features
    stage1_image_size = 512
    stage1_batch_size = 16  # A100 40GB can handle this for ConvNeXt Base
    stage1_epochs = 6
    stage1_lr = 1e-4  # Constant learning rate
    stage1_accum_iter = 1  # No accumulation needed if batch size is sufficient

    # ==========================================
    # Training Hyperparameters - Stage 2 (Fine-Grained Adaptation)
    # ==========================================
    # Fine-tuning on high resolution for lesion detection
    stage2_image_size = 1024
    stage2_batch_size = 2  # Small batch size due to memory constraints at 1024px
    stage2_epochs = 4
    stage2_lr = 1e-5  # Lower learning rate for fine-tuning
    stage2_accum_iter = 16  # Accumulate gradients to achieve effective batch size of 32

    # ==========================================
    # Optimization & Scheduling
    # ==========================================
    weight_decay = 1e-5
    max_grad_norm = 10.0
    early_stopping_patience = 3

    # ==========================================
    # Preprocessing
    # ==========================================
    # Circle crop parameters
    crop_margin = 0  # Margin for ROI cropping

    @classmethod
    def setup(cls):
        """
        Creates necessary directories for the experiment.
        """
        os.makedirs(cls.working_dir, exist_ok=True)
        os.makedirs(cls.models_dir, exist_ok=True)
        os.makedirs(cls.predictions_dir, exist_ok=True)
        os.makedirs(cls.cache_dir, exist_ok=True)

        # Print configuration summary
        print(f"Configuration Setup:")
        print(f"  Device: {cls.device}")
        print(f"  Working Directory: {cls.working_dir}")
        print(
            f"  Stage 1: {cls.stage1_image_size}px, Batch {cls.stage1_batch_size}, LR {cls.stage1_lr}"
        )
        print(
            f"  Stage 2: {cls.stage2_image_size}px, Batch {cls.stage2_batch_size}, LR {cls.stage2_lr}, Accum {cls.stage2_accum_iter}"
        )


# Initialize directories immediately upon import
Config.setup()
