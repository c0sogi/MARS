import os
import torch


class Config:
    """
    Central configuration for the Cactus Identification task using a
    Stem-Adapted Pretrained Heterogeneous Stacking Ensemble.
    """

    def __init__(
        self,
        debug: bool = False,
        epochs: int = 50,
        batch_size: int = 128,
        num_workers: int = 4,
        output_dir: str = "./working/idea_8",
    ):
        """
        Initialize configuration with flexible overrides.

        Args:
            debug (bool): If True, reduces epochs and dataset size for rapid testing.
            epochs (int): Maximum number of training epochs.
            batch_size (int): Batch size for training and inference.
            num_workers (int): Number of subprocesses for data loading.
            output_dir (str): Directory to store model checkpoints and cached data.
        """

        # --- General Settings ---
        self.debug = debug
        self.seed = 42

        # --- Paths ---
        self.input_dir = "./input"
        self.metadata_dir = "./metadata"

        # Metadata files (pre-generated)
        self.train_metadata_path = os.path.join(self.metadata_dir, "train_metadata.csv")
        self.val_metadata_path = os.path.join(self.metadata_dir, "val_metadata.csv")
        self.test_metadata_path = os.path.join(self.metadata_dir, "test_metadata.csv")

        # Output directories
        self.output_dir = output_dir
        self.submission_dir = "./submission"
        self.submission_path = os.path.join(self.submission_dir, "submission.csv")

        # Ensure working directories exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.submission_dir, exist_ok=True)

        # --- Compute ---
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_workers = num_workers
        # Optimize for fixed input size
        self.use_cudnn_benchmark = True

        # --- Data Configuration ---
        self.img_size = 32
        self.num_classes = 1
        self.batch_size = batch_size

        # Debugging: Limit dataset size if debugging
        self.debug_sample_size = 500 if self.debug else None

        # --- Model Architecture ---
        # The "Adapted Triad" of architectures
        self.model_names = [
            "seresnext50_32x4d",  # High capacity, channel attention
            "densenet121",  # Feature reuse, robust to stem mod
            "resnet34",  # Stable residual anchor
        ]
        self.pretrained = True

        # Stem Surgery Configuration
        # If True, replaces the initial 7x7 stride-2 stem with a 3x3 stride-1 conv
        # to preserve information for 32x32 inputs.
        self.stem_surgery = True

        # --- Training Hyperparameters ---
        self.n_folds = 5
        self.epochs = 5 if self.debug else epochs
        self.patience = 15  # Relaxed patience for Mixup convergence
        self.learning_rate = 1e-3
        # Increased weight decay for AdamW (Cite solution_lesson_node_00016)
        self.weight_decay = 1e-2

        # Regularization
        self.mixup_alpha = 1.0

        # --- Inference ---
        # Test-Time Augmentation (TTA)
        # Strategies: Original, Horizontal Flip, Vertical Flip
        self.tta_steps = 3

    def to_dict(self):
        """Returns a dictionary representation of the configuration for logging."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("__")}
