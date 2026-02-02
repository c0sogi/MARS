import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Ensures deterministic behavior for cuDNN.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Config:
    """
    Configuration for the Histopathology Tumor Detection task.
    Implements the strategy: Memory-Resident Homogeneous Ensemble of ConvNeXt-Tiny
    with Multi-Sample Dropout and Isotropic Augmentation.
    """

    # --- General ---
    seed = 42
    debug = False  # Set to True to run on a small subset for verification
    debug_sample_size = 1000  # Number of samples to use when debug=True

    # --- Compute ---
    num_workers = 12  # Available vCPUs
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Paths ---
    input_dir = "./input"
    metadata_dir = "./metadata"
    working_dir = "./working/idea_10"

    # Sub-directories for artifacts
    cache_dir = os.path.join(working_dir, "cache")
    checkpoints_dir = os.path.join(working_dir, "checkpoints")
    submission_dir = "./submission"

    # File Paths
    train_metadata_path = os.path.join(metadata_dir, "train.csv")
    val_metadata_path = os.path.join(metadata_dir, "val.csv")
    test_metadata_path = os.path.join(metadata_dir, "test.csv")
    submission_path = os.path.join(submission_dir, "submission.csv")

    # --- Data Strategy ---
    load_in_memory = True  # Load entire dataset into RAM to remove I/O bottlenecks

    # Dimensions
    image_raw_size = 96  # Original patch size provided in dataset
    image_crop_size = 64  # Contextual crop size for training/inference

    # Normalization (Derived from dataset analysis)
    # Mean: [0.7035, 0.5476, 0.6975], Std: [0.2388, 0.2821, 0.2159]
    normalize_mean = [0.7035, 0.5476, 0.6975]
    normalize_std = [0.2388, 0.2821, 0.2159]

    # Augmentation
    aug_rotate_limit = 180  # Continuous rotation (-180 to 180 degrees)
    aug_flip_prob = 0.5  # Probability for Horizontal and Vertical flips
    aug_color_jitter_prob = 1.0  # Apply intensity invariance to 100% of images
    aug_color_jitter_strength = 0.2  # Strength for brightness/contrast/sat/hue

    # --- Model Architecture ---
    model_name = "convnext_tiny"
    pretrained = True
    in_channels = 3
    num_classes = 1

    # Multi-Sample Dropout Configuration
    use_multi_sample_dropout = True
    multi_sample_dropout_count = 5
    multi_sample_dropout_rate = 0.2

    # --- Training Hyperparameters ---
    n_folds = 5
    epochs = 15
    batch_size = 256  # Sized for A100-40GB GPU

    # Optimization
    learning_rate = 2e-4  # Lower LR for stability (Cite solution_lesson_node_00003)
    min_lr = 1e-6
    weight_decay = 0.05

    # Regularization
    mixup_alpha = 0.2

    # Exponential Moving Average (EMA)
    use_ema = True
    ema_decay = 0.995  # Tuned for 15 epochs (Cite solution_lesson_node_00024)

    # --- Inference ---
    tta_steps = 8  # 8-view Test Time Augmentation (Dihedral Group)

    @classmethod
    def setup(cls):
        """
        Creates necessary directories and sets random seeds.
        """
        os.makedirs(cls.working_dir, exist_ok=True)
        os.makedirs(cls.cache_dir, exist_ok=True)
        os.makedirs(cls.checkpoints_dir, exist_ok=True)
        os.makedirs(cls.submission_dir, exist_ok=True)

        seed_everything(cls.seed)
