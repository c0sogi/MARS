import os
import torch


class CFG:
    """
    Configuration class for Cassava Leaf Disease Classification.
    Encapsulates hyperparameters, file paths, and model settings.
    """

    # --- General Settings ---
    debug = False  # Set to True for fast debugging on small subset
    seed = 42  # Random seed for reproducibility
    num_workers = 4  # Number of DataLoader workers
    print_freq = 100  # Logging frequency

    # --- Compute ---
    # Automatically detect GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Data Configuration ---
    image_size = (
        512  # Resolution scaled up for fine-grained details (Cite Lesson 00005)
    )
    target_size = 5  # Number of disease classes
    target_col = "label"  # Name of target column in CSV

    # --- Paths ---
    # Input directories
    input_root = "./input"
    train_images_dir = os.path.join(input_root, "train_images")
    test_images_dir = os.path.join(input_root, "test_images")

    # Metadata paths (Pre-generated in ./metadata)
    train_csv = "./metadata/train.csv"
    val_csv = "./metadata/val.csv"
    test_csv = "./metadata/test.csv"
    sample_sub = os.path.join(input_root, "sample_submission.csv")

    # Output directory for artifacts (Model checkpoints, logs)
    output_dir = "./working/idea_2"
    model_save_name = "best_model.pth"

    # --- Model Architecture ---
    model_name = "tf_efficientnet_b4_ns"  # EfficientNet-B4 with Noisy Student weights
    pretrained = True
    drop_rate = 0.3  # Dropout rate for the head
    drop_path_rate = 0.2  # Stochastic depth rate

    # --- Training Hyperparameters ---
    epochs = 25  # Extended training for strong augmentation (Cite Lesson 00009)
    batch_size = 32  # Suitable for A100 40GB with B4@380

    # Regularization
    label_smoothing = 0.1  # Label smoothing factor (Cite Lesson 00008)

    # Optimizer
    lr = 1e-4  # Initial learning rate
    min_lr = 1e-6  # Minimum learning rate for scheduler
    weight_decay = 1e-6  # Weight decay

    # Scheduler
    scheduler = "CosineAnnealingLR"  # Learning rate scheduler
    T_max = epochs  # Cycle length for CosineAnnealing

    # Strategy
    freeze_epochs = 1  # Freeze backbone for the first epoch (Warmup)

    # --- Augmentation & Regularization ---
    # MixUp / CutMix settings
    mix_p = 1.0  # Probability of applying MixUp or CutMix
    mix_alpha = 0.4  # Beta distribution parameter for mixing

    # --- Inference ---
    tta = True  # Enable Test Time Augmentation
    tta_steps = 3  # Number of TTA steps (e.g., Original + HFlip + VFlip)

    @staticmethod
    def setup():
        """
        Ensures the output directory exists.
        """
        os.makedirs(CFG.output_dir, exist_ok=True)


# Execute setup on import to guarantee directory existence
CFG.setup()
