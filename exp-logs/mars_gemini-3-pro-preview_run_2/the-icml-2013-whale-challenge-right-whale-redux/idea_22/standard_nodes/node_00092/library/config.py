import os
import torch
import numpy as np
import random


class Config:
    """
    Configuration class for Right Whale Call Detection.
    Implements the 'Uncertainty-Aware Self-Distilled Heterogeneous Ensemble' strategy.
    """

    # --- General System Settings ---
    seed = 42
    debug = False  # Set to True to run on a small subset
    debug_sample_size = 100  # Number of samples to use in debug mode
    num_workers = 12  # Utilizing available vCPUs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Audio Preprocessing (Golden Recipe) ---
    sample_rate = 2000  # Native sampling rate of the dataset
    duration = 2.0  # Fixed duration in seconds
    n_fft = 1024  # High frequency resolution
    hop_length = 64  # High time resolution
    n_mels = 128  # Number of Mel bands
    fmin = 0
    fmax = None
    top_db = 80.0  # Dynamic range clamping

    # --- Augmentation ---
    spec_augment = True
    freq_mask_param = 20  # Aggressive frequency masking
    time_mask_param = 10
    mixup = False  # Explicitly excluded

    # --- Model Architecture ---
    # Heterogeneous ensemble components
    model_names = ["tf_efficientnet_b0_ns", "resnet34"]
    num_classes = 1
    pretrained = True

    # --- Training Hyperparameters ---
    epochs = 20
    batch_size = 64
    learning_rate = 1e-4
    weight_decay = 1e-6
    min_lr = 1e-6
    patience = 5  # Early stopping patience

    # --- Cross Validation ---
    n_folds = 5

    # --- Paths ---
    input_root = "./input"
    train_dir = os.path.join(input_root, "train2")
    test_dir = os.path.join(input_root, "test2")

    metadata_root = "./metadata"
    train_csv = os.path.join(metadata_root, "train.csv")
    val_csv = os.path.join(metadata_root, "val.csv")
    test_csv = os.path.join(metadata_root, "test.csv")
    sample_submission = os.path.join(input_root, "sampleSubmission.csv")

    # Working Directory for Checkpoints and Cache
    working_dir = "./working/idea_22"

    # Cache File Paths (Parquet/NPY)
    # Using specific names to avoid conflicts
    train_cache_file = os.path.join(working_dir, "cached_train_mels.npy")
    val_cache_file = os.path.join(working_dir, "cached_val_mels.npy")
    test_cache_file = os.path.join(working_dir, "cached_test_mels.npy")

    # Pseudo-labeling
    pseudo_label_confidence_threshold = 0.95  # High confidence for self-training
    pseudo_label_uncertainty_threshold = 0.05  # Low variance requirement

    @classmethod
    def setup(cls):
        """
        Initializes the environment:
        1. Creates necessary working directories.
        2. Sets random seeds for reproducibility.
        """
        # Create working directory
        os.makedirs(cls.working_dir, exist_ok=True)

        # Set seeds
        random.seed(cls.seed)
        np.random.seed(cls.seed)
        torch.manual_seed(cls.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(cls.seed)
            torch.cuda.manual_seed_all(cls.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        print(f"Configuration initialized. Working directory: {cls.working_dir}")
        print(f"Device: {cls.device}")
