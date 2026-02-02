import os
import torch


class CFG:
    """
    Configuration for Tri-Modal Heterogeneous Stacking with Intra-Fold Model Soups.
    """

    # ==========================
    # General Settings
    # ==========================
    seed = 42
    debug = False  # Set to True to run on a small subset for debugging
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_freq = 50

    # ==========================
    # Directory Paths
    # ==========================
    input_dir = "./input"
    metadata_dir = "./metadata"

    # Metadata CSVs
    train_csv = os.path.join(metadata_dir, "train.csv")
    val_csv = os.path.join(metadata_dir, "val.csv")
    test_csv = os.path.join(metadata_dir, "test.csv")

    # Output Directory (Idea 15)
    output_dir = "./working/idea_15"

    # ==========================
    # Data & Augmentation
    # ==========================
    image_size = 224

    # Augmentation parameters
    crop_scale = (0.8, 1.0)  # Minimum scale 0.8 to preserve subject
    mixup_alpha = 0.2
    cutmix_alpha = 1.0
    mixup_prob = 0.5  # Probability to apply Mixup or Cutmix

    # ==========================
    # Model Architectures
    # ==========================
    # 1. ConvNeXt Small (Convolutional)
    # 2. Swin Transformer Small (Hierarchical Transformer)
    # 3. EfficientNetV2 Small (MBConv / Squeeze-and-Excitation)
    model_names = [
        "convnext_small.fb_in22k",
        "swin_small_patch4_window7_224.ms_in22k",
        "tf_efficientnetv2_s.in21k",
    ]

    # ==========================
    # Training Hyperparameters
    # ==========================
    num_folds = 5
    epochs = 20
    batch_size = 64

    # Optimization
    learning_rate = 1e-4
    min_lr = 1e-6
    weight_decay = 0.05

    # Scheduler
    scheduler_type = "CosineAnnealingLR"

    # Early Stopping is disabled to allow models to converge fully for Soup creation
    early_stopping = False

    # ==========================
    # Model Soup Strategy
    # ==========================
    # Indices of the last 3 epochs to average (0-indexed)
    # For 20 epochs (0-19), these are 17, 18, 19
    soup_epoch_indices = [17, 18, 19]

    # ==========================
    # Inference & Stacking
    # ==========================
    tta = True  # Enable Horizontal Flip TTA

    @classmethod
    def setup(cls):
        """Ensures the output directory exists."""
        os.makedirs(cls.output_dir, exist_ok=True)


# Initialize directories
CFG.setup()
