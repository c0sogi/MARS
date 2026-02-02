import os
import torch
import random
import numpy as np


class Config:
    """
    Configuration class for the DeBERTa-v3-base Sentiment Extraction task.
    Centralizes hyperparameters, paths, and reproducibility settings.
    """

    def __init__(self, debug=False, epochs=5, n_folds=5, train_batch_size=32):
        # General Settings
        self.seed = 42
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_workers = 4  # Optimized for 12 vCPUs

        # Paths
        # Using metadata paths as per instructions
        self.train_path = "./metadata/train.csv"
        self.val_path = "./metadata/val.csv"
        self.test_path = "./metadata/test.csv"
        self.sample_submission_path = "./input/sample_submission.csv"

        # Output Directory
        self.output_dir = "./working/idea_3/"
        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Model Architecture
        self.model_name = "microsoft/deberta-v3-base"
        self.max_len = 128  # Sufficient for tweet length (max ~140 chars)
        self.dropout = 0.1

        # Training Hyperparameters
        self.epochs = epochs
        self.train_batch_size = train_batch_size
        self.valid_batch_size = 64
        self.learning_rate = 2e-5
        self.weight_decay = 0.01
        self.scheduler = "linear"
        self.warmup_ratio = 0.1
        self.label_smoothing = 0.1
        self.clip_grad_norm = 1.0

        # Cross-Validation
        self.n_folds = n_folds

        # Debugging / Runtime Control
        self.debug = debug
        self.debug_sample_size = 100  # Number of samples to use when debug=True

        # Adjust settings if in debug mode
        if self.debug:
            self.epochs = 2
            self.n_folds = 2
            self.train_batch_size = 8
            self.valid_batch_size = 8
            print(
                f"Debug mode enabled: epochs={self.epochs}, n_folds={self.n_folds}, sample_size={self.debug_sample_size}"
            )

    def seed_everything(self):
        """
        Sets the random seed for all relevant libraries to ensure reproducibility.
        """
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def get_model_path(self, fold):
        """Returns the file path for saving/loading the model for a specific fold."""
        return os.path.join(self.output_dir, f"model_fold_{fold}.pth")
