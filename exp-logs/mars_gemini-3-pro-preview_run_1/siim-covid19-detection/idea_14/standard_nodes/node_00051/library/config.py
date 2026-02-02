import os
import torch
import random
import numpy as np


class Config:
    """
    Configuration class for the ResNet18-D Multi-Task U-Net training pipeline.
    Centralizes all hyperparameters, file paths, and system settings.
    """

    def __init__(self):
        # =================================================================
        # System & Reproducibility
        # =================================================================
        self.seed = 42
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_workers = 4  # Optimized for 12 vCPUs

        # =================================================================
        # File Paths & Directories
        # =================================================================
        # Input Data
        self.input_dir = "./input"
        self.train_metadata = "./metadata/train.csv"
        self.val_metadata = "./metadata/val.csv"
        self.test_metadata = "./metadata/test.csv"
        self.sample_submission = "./input/sample_submission.csv"

        # Output & Caching
        # Using idea_14 as the designated working directory for this experiment
        self.working_dir = "./working/idea_14"
        self.output_dir = self.working_dir
        self.cache_dir = self.working_dir
        self.model_save_path = os.path.join(self.output_dir, "best_model.pth")
        self.submission_path = "./submission/submission.csv"

        # Create necessary directories
        os.makedirs(self.working_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.submission_path), exist_ok=True)

        # =================================================================
        # Data Hyperparameters
        # =================================================================
        self.image_size = 512
        self.batch_size = 32  # Conservative batch size for A100 stability with 512x512

        # Class Definitions
        self.study_label_cols = [
            "Negative for Pneumonia",
            "Typical Appearance",
            "Indeterminate Appearance",
            "Atypical Appearance",
        ]
        self.num_study_classes = len(self.study_label_cols)
        self.num_seg_classes = 1  # Opacity

        # Augmentation Settings
        # CoarseDropout settings to force global context learning
        self.aug_dropout_prob = 0.5
        self.aug_dropout_holes = 8
        self.aug_dropout_size = int(self.image_size * 0.1)  # ~10% of image size

        # =================================================================
        # Model Architecture
        # =================================================================
        self.backbone = "resnet18d"  # ResNet18-D (Deep Stem)
        self.pretrained = True
        self.decoder_channels = [256, 128, 64, 32, 16]

        # =================================================================
        # Training & Optimization
        # =================================================================
        self.epochs = 20
        self.learning_rate = 1e-4  # Conservative LR to prevent overfitting
        self.weight_decay = 1e-5

        # Scheduler (Cosine Annealing)
        self.T_max = self.epochs
        self.min_lr = 1e-6

        # Loss Weighting (1:10 ratio per strategy)
        self.study_loss_weight = 1.0
        self.image_loss_weight = 10.0

        # Validation
        self.iou_threshold = 0.5  # For mAP calculation

        # =================================================================
        # Inference
        # =================================================================
        self.tta_steps = 2  # Original + Horizontal Flip

        # Setup
        self._setup_reproducibility()

    def _setup_reproducibility(self):
        """Sets random seeds for reproducibility."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Instantiate config for easy import
cfg = Config()
