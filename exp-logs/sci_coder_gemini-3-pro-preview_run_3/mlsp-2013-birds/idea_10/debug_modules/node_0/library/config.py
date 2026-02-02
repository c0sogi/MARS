import os


class Config:
    """
    Configuration class for the Bird Species Classification Task.
    Encapsulates hyperparameters, file paths, and model settings.
    """

    def __init__(self, debug: bool = False, epochs: int = 50, batch_size: int = 16):
        """
        Initialize configuration with optional overrides for debugging and training control.

        Args:
            debug (bool): If True, runs in debug mode with a small data subset and fewer epochs.
            epochs (int): Number of training epochs. Overridden to 2 if debug is True.
            batch_size (int): Batch size for training and inference.
        """
        # =========================================================================
        # General Settings
        # =========================================================================
        self.seed = 42
        self.num_workers = (
            4 if not debug else 0
        )  # Reduce workers in debug to avoid overhead
        self.device = (
            "cuda"  # Will be handled by training script, but good to define preference
        )

        # =========================================================================
        # File Paths
        # =========================================================================
        self.input_dir = "./input"
        self.metadata_dir = "./metadata"

        # Source data paths
        self.spectrogram_dir = os.path.join(
            self.input_dir, "supplemental_data", "spectrograms"
        )
        self.train_metadata_path = os.path.join(self.metadata_dir, "train.csv")
        self.val_metadata_path = os.path.join(self.metadata_dir, "val.csv")
        self.test_metadata_path = os.path.join(self.metadata_dir, "test.csv")

        # Output paths
        self.working_dir = "./working/idea_10"
        self.submission_dir = "./submission"
        self.submission_path = os.path.join(self.submission_dir, "submission.csv")

        # Ensure output directories exist
        os.makedirs(self.working_dir, exist_ok=True)
        os.makedirs(self.submission_dir, exist_ok=True)

        # =========================================================================
        # Data Configuration
        # =========================================================================
        # Resizing to 224x224 as per strategy (preserving global context)
        self.image_size = (224, 224)

        # Input channels must be 3 for pretrained ImageNet models.
        # Single-channel spectrograms will be replicated.
        self.input_channels = 3

        self.num_classes = 19

        # Debugging constraints
        self.debug = debug
        self.debug_sample_size = 32 if debug else None

        # =========================================================================
        # Model Architecture
        # =========================================================================
        # Dual-backbone ensemble strategy
        self.models = ["resnet18", "efficientnet_b0"]
        self.pretrained = True

        # =========================================================================
        # Training Hyperparameters
        # =========================================================================
        self.n_folds = 5
        self.batch_size = batch_size
        self.epochs = epochs if not debug else 2

        # Optimizer settings (AdamW)
        self.learning_rate = 1e-3
        self.weight_decay = 1e-2

        # Scheduler settings (Cosine Annealing)
        self.eta_min = 1e-6

        # =========================================================================
        # Loss Function (Asymmetric Loss)
        # =========================================================================
        # Parameters to handle high class imbalance
        self.asl_gamma_neg = 4.0
        self.asl_gamma_pos = 1.0
        self.asl_clip = 0.05

        # =========================================================================
        # Augmentation Strategy
        # =========================================================================
        self.mixup_alpha = 0.4
        self.time_shift_limit = 0.2  # Fraction of total width
        self.brightness_limit = 0.2
        self.contrast_limit = 0.2
        # Horizontal flip is strictly disabled due to temporal causality of audio

    def __repr__(self):
        """Pretty print configuration."""
        return "\n".join([f"{k}: {v}" for k, v in self.__dict__.items()])
