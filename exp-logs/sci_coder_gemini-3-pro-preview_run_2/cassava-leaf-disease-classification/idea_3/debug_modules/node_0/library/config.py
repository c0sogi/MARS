import os
import torch


class CFG:
    """
    Configuration class for the Cassava Leaf Disease Classification experiment.
    Centralizes hyperparameters, paths, and settings.
    """

    # ====================================================
    # General Settings
    # ====================================================
    seed = 42
    debug = False  # Set to True for fast debugging runs
    num_workers = 4  # Optimized for 12 vCPUs
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ====================================================
    # Data Paths
    # ====================================================
    input_root = "./input"
    train_csv = "./metadata/train.csv"
    val_csv = "./metadata/val.csv"
    test_csv = "./metadata/test.csv"

    # Output directory for artifacts (checkpoints, logs, cache)
    output_dir = "./working/idea_3"

    # ====================================================
    # Model Architecture
    # ====================================================
    model_name = "convnext_tiny"
    pretrained = True
    num_classes = 5
    image_size = 224  # Standard resolution for ConvNeXt

    # ====================================================
    # Training Hyperparameters
    # ====================================================
    epochs = 10
    batch_size = 32  # Small batch size for gradient noise regularization
    learning_rate = 1e-4
    weight_decay = 1e-2
    min_lr = 1e-6  # For Cosine Annealing

    # ====================================================
    # Regularization
    # ====================================================
    label_smoothing = 0.1  # Prevents overconfidence on noisy labels
    drop_path_rate = 0.1  # Stochastic Depth rate for ConvNeXt

    # ====================================================
    # Inference
    # ====================================================
    tta = True  # Enable Test Time Augmentation (Horizontal Flip)

    # ====================================================
    # Debugging / Dataset Control
    # ====================================================
    # If debug is True or these are set, dataset will be truncated
    train_subset_size = None
    val_subset_size = None

    @classmethod
    def setup(cls):
        """
        Ensures the output directory exists.
        This is critical for caching mechanisms that rely on this path.
        """
        os.makedirs(cls.output_dir, exist_ok=True)


# Execute setup upon import to guarantee directory existence
CFG.setup()
