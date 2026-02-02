import os
import sys
import random
import numpy as np
import torch


class Config:
    """
    Configuration class for Cassava Leaf Disease Classification.
    Centralizes hyperparameters, file paths, and system settings for the
    ConvNeXt-Small + SWA pipeline.
    """

    def __init__(self, debug: bool = False):
        # --- General Settings ---
        self.seed = 42
        self.debug = debug
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_workers = 12  # Utilizing available vCPUs

        # --- Directories & Paths ---
        self.input_root = "./input"
        self.metadata_dir = "./metadata"
        self.working_dir = "./working/idea_5"
        self.submission_dir = "./submission"

        # Input Metadata (Pre-generated)
        self.train_metadata_path = os.path.join(self.metadata_dir, "train.csv")
        self.val_metadata_path = os.path.join(self.metadata_dir, "val.csv")
        self.test_metadata_path = os.path.join(self.metadata_dir, "test.csv")

        # Output Paths
        self.output_dir = self.working_dir
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        self.submission_path = os.path.join(self.submission_dir, "submission.csv")

        # Ensure working directories exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.submission_dir, exist_ok=True)

        # --- Data Configuration ---
        self.num_classes = 5
        # Progressive Resizing Resolutions
        self.input_size_stage1 = 384
        self.input_size_stage2 = 512

        # Debugging: Use a subset of data if debug is True
        self.data_subset_fraction = 0.1 if self.debug else 1.0

        # --- Model Architecture ---
        # Using ConvNeXt-Small pretrained on ImageNet-21k
        # timm model name
        self.model_name = "convnext_small.fb_in22k"
        self.drop_path_rate = 0.0  # Disabled to maximize capacity with heavy augs
        self.dropout_rate = 0.0
        self.head_dropout_rate = 0.5
        self.use_multi_sample_dropout = True
        self.multi_sample_dropout_count = 5

        # --- Training Hyperparameters ---
        self.batch_size = 32  # Effective batch size for A100
        self.accum_iter = 1  # Gradient accumulation steps
        self.max_grad_norm = 1000.0
        self.weight_decay = 1e-4
        self.early_stopping_patience = 5

        # Epochs & Schedule
        if self.debug:
            self.epochs_stage1 = 1
            self.epochs_stage2 = 1
            self.epochs_swa = 1
        else:
            self.epochs_stage1 = 10
            self.epochs_stage2 = 5
            self.epochs_swa = 5

        # Learning Rates
        self.lr_stage1 = 1e-4
        self.min_lr_stage1 = 1e-6
        self.warmup_epochs_stage1 = 1

        self.lr_stage2 = 5e-5
        self.min_lr_stage2 = 1e-6
        self.warmup_epochs_stage2 = 0  # No warmup for fine-tuning stage

        self.swa_lr = 5e-5  # Constant LR for SWA phase

        # --- Augmentation ---
        self.mixup_alpha = 0.8
        self.cutmix_alpha = 1.0
        self.mixup_prob = 1.0
        self.mixup_switch_prob = 0.5
        self.mixup_mode = "batch"
        self.random_resized_crop_scale = (0.3, 1.0)
        self.label_smoothing = 0.1

        # --- Inference ---
        self.tta_steps = 3  # Original + HFlip + VFlip

    def seed_everything(self):
        """
        Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
        """
        random.seed(self.seed)
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        # Deterministic behavior for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def __str__(self):
        """Returns a string representation of the config for logging."""
        return str(self.__dict__)
