import os
import torch


class Config:
    """
    Configuration class for Cassava Leaf Disease Classification.
    Implements parameters for Progressive Resolution Training with ConvNeXt Small.
    """

    def __init__(self, debug: bool = False):
        """
        Initialize configuration.

        Args:
            debug (bool): If True, runs in debug mode with fewer epochs and data samples.
        """
        self.debug = debug

        # General Settings
        self.seed = 42
        self.num_workers = 12
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.print_freq = 100

        # Directory Paths
        self.input_dir = "./input"
        self.metadata_dir = "./metadata"
        self.train_metadata_path = os.path.join(self.metadata_dir, "train.csv")
        self.val_metadata_path = os.path.join(self.metadata_dir, "val.csv")
        self.test_metadata_path = os.path.join(self.metadata_dir, "test.csv")

        # Output/Working Directory (Idea 7 specific)
        self.working_dir = "./working/idea_7"
        self.submission_dir = "./submission"
        self.submission_path = os.path.join(self.submission_dir, "submission.csv")

        # Create necessary directories
        os.makedirs(self.working_dir, exist_ok=True)
        os.makedirs(self.submission_dir, exist_ok=True)

        # Model Architecture
        # Using ConvNeXt Small pretrained on ImageNet-1k
        self.model_name = "convnext_small.fb_in1k"
        self.num_classes = 5
        self.drop_path_rate = 0.4  # Stochastic Depth rate
        self.dropout_rate = 0.0  # Classifier dropout

        # Training Strategy: 5-Fold Stratified CV
        self.n_folds = 5
        self.batch_size = 32
        self.accumulate_grad_batches = 1

        # Progressive Resolution Schedule
        # Phase 1: Coarse Learning (Lower Resolution)
        # Phase 2: Fine Tuning (Higher Resolution)
        if self.debug:
            self.phase1_epochs = 1
            self.phase2_epochs = 1
            self.subset_size = 100  # Only use 100 samples in debug
        else:
            self.phase1_epochs = 12
            self.phase2_epochs = 8
            self.subset_size = None  # Use full dataset

        self.phase1_image_size = 224
        self.phase2_image_size = 384

        # Optimizer (AdamW) & Scheduler (Cosine Annealing)
        self.lr = 1e-4  # Base learning rate
        self.min_lr = 1e-6
        self.weight_decay = 0.05
        self.warmup_epochs = 1  # Warmup for phase 1

        # Augmentation & Regularization
        # MixUp and CutMix
        self.mixup_p = 0.5  # Probability of applying MixUp/CutMix
        self.mixup_alpha = 0.8  # MixUp beta distribution parameter
        self.cutmix_alpha = 1.0  # CutMix beta distribution parameter
        self.label_smoothing = 0.1  # Used when MixUp/CutMix is not applied (if any)

        # Inference
        self.tta = True  # Test Time Augmentation (Horizontal Flip)

    def get_total_epochs(self):
        return self.phase1_epochs + self.phase2_epochs

    def __repr__(self):
        return (
            f"Config(debug={self.debug}, model={self.model_name}, "
            f"folds={self.n_folds}, batch_size={self.batch_size}, "
            f"phase1={self.phase1_epochs}ep@{self.phase1_image_size}, "
            f"phase2={self.phase2_epochs}ep@{self.phase2_image_size})"
        )
