import os
import torch


class Config:
    """
    Central configuration for the Artwork Attribute Labeling task.
    Aligns with the strategy: ConvNeXt-Tiny, 320x320, GeM Pooling, and weighted BCE loss.
    """

    # =======================
    # General Settings
    # =======================
    seed = 42
    debug = False  # Set to True to run on a small subset for debugging
    debug_sample_size = 2000  # Number of samples to use in debug mode
    num_workers = 4  # Optimized for the available vCPUs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =======================
    # Data Paths
    # =======================
    # Root directory for input images
    input_root = "./input"

    # Pre-generated metadata files
    train_metadata_path = "./metadata/train.csv"
    val_metadata_path = "./metadata/val.csv"
    test_metadata_path = "./metadata/test.csv"

    # Label descriptions
    labels_path = "./input/labels.csv"

    # =======================
    # Model Architecture
    # =======================
    model_name = "convnext_tiny"  # timm backbone
    pretrained = True
    image_size = 320  # Increased resolution for fine detail
    num_classes = 3474  # Total unique attributes

    # Pooling strategy
    pooling_type = "gem"  # Generalized Mean Pooling
    gem_p = 3.0  # Initial power for GeM pooling (learnable)

    # =======================
    # Training Hyperparameters
    # =======================
    epochs = 10  # Sufficient for convergence with pre-trained backbone
    batch_size = 64  # Fits 320x320 ConvNeXt-Tiny on A100 (40GB)

    # Optimizer (AdamW)
    learning_rate = 2e-4
    weight_decay = 1e-2

    # Scheduler (Cosine Annealing)
    min_lr = 1e-6
    T_max = epochs  # Cycle length matches epochs

    # =======================
    # Loss Function Configuration
    # =======================
    # BCEWithLogitsLoss settings
    pos_weight = 12.0  # Addresses high class imbalance/sparsity
    label_smoothing = 0.05  # Mitigates noise in unverified annotations

    # =======================
    # Output & Logging
    # =======================
    # Directory for saving model checkpoints and cached data
    output_dir = "./working/idea_2"

    # Path to save the best model
    model_save_path = os.path.join(output_dir, "convnext_tiny_best.pth")

    # Path for the final submission file
    submission_dir = "./submission"
    submission_path = os.path.join(submission_dir, "submission.csv")

    @classmethod
    def setup(cls):
        """
        Ensures necessary output directories exist.
        """
        os.makedirs(cls.output_dir, exist_ok=True)
        os.makedirs(cls.submission_dir, exist_ok=True)

        # Print configuration summary
        print(f"Config Setup Complete:")
        print(f"  Device: {cls.device}")
        print(f"  Model: {cls.model_name} (Input: {cls.image_size}x{cls.image_size})")
        print(f"  Output Dir: {cls.output_dir}")
        print(f"  Debug Mode: {cls.debug}")
