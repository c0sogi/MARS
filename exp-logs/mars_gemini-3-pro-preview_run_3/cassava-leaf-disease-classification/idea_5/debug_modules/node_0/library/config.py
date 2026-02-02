import os
import torch


class CFG:
    """
    Configuration class for Cassava Leaf Disease Classification.
    Centralizes settings for the CoAtNet-2 Hybrid Model pipeline.
    """

    # ====================================================
    # General Settings
    # ====================================================
    debug = False  # Set to True to run on a small subset for debugging
    seed = 42  # Fixed seed for deterministic results
    num_workers = 4  # Number of CPU workers for data loading
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ====================================================
    # Data Configuration
    # ====================================================
    # Metadata files generated in ./metadata
    train_csv = "./metadata/train.csv"
    val_csv = "./metadata/val.csv"
    test_csv = "./metadata/test.csv"

    # Image directories
    train_root = "./input/train_images"
    test_root = "./input/test_images"

    # Input specifications
    num_classes = 5
    img_size = 384  # Resizing images to 384x384 for high-fidelity features

    # ====================================================
    # Model Architecture
    # ====================================================
    # CoAtNet-2: Composite Attention Network (Hybrid CNN + Transformer)
    # Using ImageNet-21k pre-trained weights.
    # Note: Native resolution is 224, but we fine-tune at 384.
    model_name = "coatnet_2_rw_224.sw_in12k"

    # ====================================================
    # Training Hyperparameters
    # ====================================================
    epochs = 10  # Total training epochs (fast convergence expected)
    train_batch_size = 32  # Optimized for A100 40GB with 75M params @ 384x384
    valid_batch_size = 64  # Larger batch size for validation

    # Optimizer settings (AdamW)
    lr = 1e-4  # Initial learning rate
    weight_decay = 1e-4  # Weight decay for regularization

    # Scheduler settings (Cosine Annealing)
    scheduler = "CosineAnnealingLR"
    min_lr = 1e-6  # Minimum learning rate after annealing
    T_max = epochs  # Cycle length matching total epochs

    # Loss Function
    label_smoothing = 0.1  # Label smoothing factor to prevent overfitting

    # Early Stopping
    early_stopping_patience = (
        3  # Stop if validation accuracy doesn't improve for 3 epochs
    )

    # ====================================================
    # Augmentation & Regularization
    # ====================================================
    # MixUp and CutMix parameters
    mixup_alpha = 0.2
    cutmix_alpha = 1.0
    mixup_prob = 0.5  # Probability to apply MixUp/CutMix

    # ====================================================
    # Inference / Submission
    # ====================================================
    tta_steps = 3  # Number of TTA steps (Original + Flips)

    # ====================================================
    # Output Directory
    # ====================================================
    output_dir = "./working/idea_5"

    @classmethod
    def setup(cls):
        """Creates the output directory if it does not exist."""
        os.makedirs(cls.output_dir, exist_ok=True)


# Ensure output directory exists upon import
CFG.setup()
