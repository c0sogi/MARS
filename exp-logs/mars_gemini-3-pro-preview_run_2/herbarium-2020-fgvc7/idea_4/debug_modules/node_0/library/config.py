import os
import random
import numpy as np
import torch


class Config:
    """
    Configuration class for the Plant Species Classification task.
    Centralizes parameters for data loading, model architecture, training, and paths.
    """

    def __init__(self, debug=False, epochs=15, batch_size=128, image_size=224):
        """
        Initialize configuration.

        Args:
            debug (bool): If True, runs in debug mode with smaller dataset and fewer epochs.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            image_size (int): Input image size (height and width).
        """
        # General Settings
        self.seed = 42
        self.debug = debug
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Directories
        self.input_dir = "./input"
        self.metadata_dir = "./metadata"
        self.working_dir = "./working/idea_4"
        self.submission_dir = "./submission"

        # Ensure output directories exist
        os.makedirs(self.working_dir, exist_ok=True)
        os.makedirs(self.submission_dir, exist_ok=True)

        # File Paths
        self.train_csv_path = os.path.join(self.metadata_dir, "train.csv")
        self.val_csv_path = os.path.join(self.metadata_dir, "val.csv")
        self.test_csv_path = os.path.join(self.metadata_dir, "test.csv")
        self.submission_path = os.path.join(self.submission_dir, "submission.csv")

        # Model Artifact Paths
        self.best_model_path = os.path.join(self.working_dir, "best_model.pth")
        self.checkpoint_path = os.path.join(self.working_dir, "checkpoint.pth")
        self.log_path = os.path.join(self.working_dir, "train.log")

        # Data Parameters
        self.num_classes = 32093
        self.img_size = image_size
        self.num_workers = 12

        # Model Architecture
        self.model_name = "swin_tiny_patch4_window7_224"
        self.pretrained = True
        self.drop_path_rate = 0.1  # Stochastic depth rate

        # Training Hyperparameters
        self.epochs = 2 if self.debug else epochs
        self.batch_size = 32 if self.debug else batch_size
        self.learning_rate = 1e-4
        self.weight_decay = 0.05
        self.patience = 3  # Early stopping patience

        # Debugging
        # If debug is True, limit the number of samples to process
        self.debug_sample_size = 2000 if self.debug else None

    def seed_everything(self):
        """
        Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
        """
        random.seed(self.seed)
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
