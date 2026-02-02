import os
import torch


class Config:
    """
    Configuration class for the Apple Disease Detection pipeline.
    Encapsulates all hyperparameters, file paths, and model settings.
    """

    def __init__(self, debug: bool = False, epochs: int = 25, batch_size: int = 32):
        """
        Initialize the configuration.

        Args:
            debug (bool): If True, runs the pipeline on a small subset of data for debugging.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training and inference.
        """
        # ==========================================
        # General Settings
        # ==========================================
        self.seed = 42
        self.debug = debug
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Use available CPUs but cap at a reasonable number for dataloaders
        self.num_workers = min(12, os.cpu_count() if os.cpu_count() else 4)

        # ==========================================
        # Directories & Paths
        # ==========================================
        self.input_dir = "./input"
        self.metadata_dir = "./metadata"

        # Output directories
        self.working_dir = "./working/idea_3"
        self.submission_dir = "./submission"

        # Ensure output directories exist
        os.makedirs(self.working_dir, exist_ok=True)
        os.makedirs(self.submission_dir, exist_ok=True)

        # File paths
        self.train_metadata_path = os.path.join(self.metadata_dir, "train_metadata.csv")
        self.val_metadata_path = os.path.join(self.metadata_dir, "val_metadata.csv")
        self.test_metadata_path = os.path.join(self.metadata_dir, "test_metadata.csv")
        self.sample_submission_path = os.path.join(
            self.input_dir, "sample_submission.csv"
        )
        self.submission_path = os.path.join(self.submission_dir, "submission.csv")
        self.images_dir = os.path.join(self.input_dir, "images")

        # ==========================================
        # Data Configuration
        # ==========================================
        self.img_size = 256
        self.num_classes = 4
        # Target columns in specific order matching the competition format
        self.target_cols = ["healthy", "multiple_diseases", "rust", "scab"]
        self.n_folds = 5

        # ==========================================
        # Model Configuration
        # ==========================================
        self.model_name = "resnext50_32x4d"
        self.pretrained = True

        # ==========================================
        # Training Configuration
        # ==========================================
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = 1e-4
        self.weight_decay = 1e-4
        self.early_stopping_patience = 5

        # Scheduler (Cosine Annealing Warm Restarts)
        self.T_0 = 10  # Number of iterations for the first restart
        self.T_mult = 1  # A factor increases T_i after a restart
        self.eta_min = 1e-6  # Minimum learning rate

        # Mixup Regularization
        self.use_mixup = True
        self.mixup_alpha = 0.4
