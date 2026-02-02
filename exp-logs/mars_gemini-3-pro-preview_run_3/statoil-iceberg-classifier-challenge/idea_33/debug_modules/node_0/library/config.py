import os
import torch


class Config:
    def __init__(self, debug: bool = False, epochs: int = 75, batch_size: int = 32):
        """
        Configuration for the NBHA-CNN pipeline.

        Args:
            debug (bool): If True, runs in debug mode with fewer epochs/folds and subsets data.
            epochs (int): Maximum number of training epochs.
            batch_size (int): Batch size for training.
        """
        # ---------------------------------------------------------------------
        # General Settings
        # ---------------------------------------------------------------------
        self.seed = 42
        self.debug = debug
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # 12 vCPUs available, 4 workers is a safe and efficient default
        self.num_workers = 4

        # ---------------------------------------------------------------------
        # File Paths
        # ---------------------------------------------------------------------
        # Metadata (Already generated in ./metadata)
        self.metadata_dir = "./metadata"
        self.train_meta_path = os.path.join(self.metadata_dir, "train.csv")
        self.val_meta_path = os.path.join(self.metadata_dir, "val.csv")
        self.test_meta_path = os.path.join(self.metadata_dir, "test.csv")

        # Raw Input (Read-only)
        self.input_dir = "./input"
        self.train_json_path = os.path.join(self.input_dir, "train.json")
        self.test_json_path = os.path.join(self.input_dir, "test.json")
        self.sample_submission_path = os.path.join(
            self.input_dir, "sample_submission.csv"
        )

        # Working Directory (For caching preprocessed data and checkpoints)
        self.working_dir = "./working/idea_33"
        self.cache_dir = self.working_dir
        self.checkpoint_dir = os.path.join(self.working_dir, "checkpoints")

        # Submission Directory
        self.submission_dir = "./submission"
        self.submission_path = os.path.join(self.submission_dir, "submission.csv")

        # Create necessary writable directories
        os.makedirs(self.working_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.submission_dir, exist_ok=True)

        # ---------------------------------------------------------------------
        # Data Parameters
        # ---------------------------------------------------------------------
        self.image_size = 75
        # Input is 3 channels: Band 1 (HH), Band 2 (HV), and Synthetic Avg ((HH+HV)/2)
        self.input_channels = 3
        self.missing_angle_strategy = "median"

        # ---------------------------------------------------------------------
        # Model Architecture: Non-Bottleneck Hybrid-Attentive Plain CNN (NBHA-CNN)
        # ---------------------------------------------------------------------
        self.model_name = "NBHA_CNN"

        # Backbone: Plain CNN (4 blocks)
        # Strategy: Sequential downsampling to filter speckle noise
        self.backbone_stages = 4
        self.channel_widths = [64, 128, 128, 128]
        self.use_bias = True  # Retain bias to preserve initialization dynamics
        self.leaky_relu_slope = 0.1  # Preserve semantic negative values

        # Attention: Non-Bottleneck SE Module
        # Strategy: Full-rank MLP (128->128->128) to capture global dependencies
        self.se_reduction_ratio = 1

        # Readout: Selective Hierarchical Max Pooling
        # Strategy: Extract from Stage 3 (medium-scale) and Stage 4 (abstract)
        # Indices are 0-based, so Stage 3 is index 2, Stage 4 is index 3
        self.readout_stages = [2, 3]

        # Classification Head
        self.dropout_rate = 0.5

        # ---------------------------------------------------------------------
        # Training Hyperparameters
        # ---------------------------------------------------------------------
        self.n_folds = 5
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = 12

        # Optimization
        # Strategy: AdamW with constant LR to decouple weight decay from updates
        self.optimizer_name = "AdamW"
        self.learning_rate = 1e-3
        self.weight_decay = 1e-2
        self.loss_function = "BCEWithLogitsLoss"

        # Evaluation
        self.use_tta = False  # Explicitly disable TTA

        # ---------------------------------------------------------------------
        # Debug Overrides
        # ---------------------------------------------------------------------
        if self.debug:
            self.epochs = 2
            self.n_folds = 2
            self.batch_size = 8
            # Note: Dataset subsetting should be handled by the data loader
            # checking self.debug

    def get_checkpoint_path(self, fold_idx: int) -> str:
        """Returns the path for saving/loading the model checkpoint for a specific fold."""
        return os.path.join(self.checkpoint_dir, f"model_fold_{fold_idx}.pth")

    def get_cache_path(self, filename: str) -> str:
        """Returns the full path for a cached file in the working directory."""
        return os.path.join(self.cache_dir, filename)
