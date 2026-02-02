import os
import torch


class Config:
    """
    Configuration class for the Apple Disease Detection pipeline.

    This class encapsulates all hyperparameters, file paths, and model settings
    to ensure reproducibility and centralization of control.
    """

    def __init__(self, debug: bool = False, epochs: int = 15, batch_size: int = 4):
        """
        Initialize the configuration.

        Args:
            debug (bool): If True, runs the pipeline on a small subset of data for debugging.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training and inference.
        """
        # =========================================================================
        # Global Settings
        # =========================================================================
        self.seed = 42
        self.debug = debug
        self.debug_sample_size = 50  # Number of samples to use in debug mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_workers = 4  # Optimized for 12 vCPUs

        # =========================================================================
        # Directory and File Paths
        # =========================================================================
        self.input_dir = "./input"
        self.metadata_dir = "./metadata"
        self.working_dir = "./working"

        # Specific directory for this experimental iteration (Idea 6)
        # This is used for caching and saving models
        self.idea_dir = os.path.join(self.working_dir, "idea_6")
        os.makedirs(self.idea_dir, exist_ok=True)

        # Metadata Files
        self.train_metadata_path = os.path.join(self.metadata_dir, "train_metadata.csv")
        self.val_metadata_path = os.path.join(self.metadata_dir, "val_metadata.csv")
        self.test_metadata_path = os.path.join(self.metadata_dir, "test_metadata.csv")

        # Output Files
        self.submission_path = os.path.join(self.working_dir, "submission.csv")
        self.best_model_path_format = os.path.join(
            self.idea_dir, "best_model_fold_{}.pth"
        )

        # =========================================================================
        # Model Hyperparameters
        # =========================================================================
        # Using EfficientNetV2-L as per the compound scaling strategy
        self.model_name = "tf_efficientnetv2_l"

        # Native resolution for EfficientNetV2-L is 480x480
        self.img_size = 480

        # Output dimension: 2 binary logits (Is Rust, Is Scab)
        # This supports the multi-label decomposition strategy
        self.num_classes = 2

        # =========================================================================
        # Training Hyperparameters
        # =========================================================================
        self.n_folds = 5
        self.epochs = epochs
        self.batch_size = batch_size
        self.gradient_accumulation_steps = 2
        self.label_smoothing = 0.05

        # Optimizer settings
        self.learning_rate = 3e-4
        self.min_lr = 1e-6
        self.weight_decay = 1e-6
        self.scheduler_patience = 3
        self.scheduler_factor = 0.5
        self.early_stopping_patience = 5

        # =========================================================================
        # Augmentation Parameters
        # =========================================================================
        # CoarseDropout parameters for distributed feature learning
        self.coarse_dropout_params = {
            "max_holes": 8,
            "max_height": 100,
            "max_width": 100,
            "min_holes": 1,
            "min_height": 16,
            "min_width": 16,
            "fill_value": 0,
            "p": 0.5,
        }

    def get_cache_path(self, dataset_name: str) -> str:
        """
        Generates a path for caching processed datasets.

        Args:
            dataset_name (str): Name of the dataset (e.g., 'train', 'val', 'test').

        Returns:
            str: Full path to the parquet cache file.
        """
        return os.path.join(self.idea_dir, f"{dataset_name}_cache.parquet")
