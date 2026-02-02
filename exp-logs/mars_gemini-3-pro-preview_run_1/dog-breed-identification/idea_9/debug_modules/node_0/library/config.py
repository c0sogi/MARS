import os
import torch


class Config:
    """
    Configuration class for the Dog Breed Classification task using
    a Self-Distilled SWA-Ensemble Strategy with ConvNeXt-Base.
    """

    def __init__(
        self,
        debug: bool = False,
        epochs: int = 15,
        batch_size: int = 32,
        image_size: int = 256,
        seed: int = 42,
    ):
        """
        Initialize configuration with flexible overrides.

        Args:
            debug (bool): If True, limits dataset size for rapid debugging.
            epochs (int): Number of training epochs per stage.
            batch_size (int): Batch size for training.
            image_size (int): Target input size for the model.
            seed (int): Random seed for reproducibility.
        """
        # --- Environment ---
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_workers = 4  # Optimized for the available vCPUs

        # --- Paths ---
        self.input_dir = "./input"
        self.metadata_dir = "./metadata"
        self.working_dir = "./working/idea_9"
        self.submission_dir = "./submission"
        self.submission_path = os.path.join(self.submission_dir, "submission.csv")

        # Ensure necessary directories exist
        os.makedirs(self.working_dir, exist_ok=True)
        os.makedirs(self.submission_dir, exist_ok=True)

        # --- Data Configuration ---
        self.debug = debug
        self.image_size = image_size
        # Resize larger than crop to preserve aspect ratio features before cropping
        self.resize_size = 274
        self.batch_size = batch_size
        self.num_classes = 120

        # --- Model Configuration ---
        # Using ConvNeXt-Base initialized with ImageNet-1k weights.
        # Explicitly avoiding 22k weights to prevent domain misalignment.
        self.model_name = "convnext_base"
        self.pretrained = True
        self.drop_path_rate = 0.1  # Stochastic depth rate
        self.head_dropout = 0.5  # Regularization for the classification head

        # --- Optimization Configuration ---
        self.epochs = epochs
        # Discriminative Learning Rates: Lower for backbone, higher for head
        self.lr_backbone = 1e-6
        self.lr_head = 1e-4
        self.weight_decay = 1e-4
        self.min_lr = 1e-7

        # --- SWA (Stochastic Weight Averaging) Configuration ---
        self.use_swa = True
        # Start SWA late in training (e.g., last 40% of epochs)
        self.swa_start_epoch = max(1, int(epochs * 0.6))
        self.swa_lr = 1e-5

        # --- Knowledge Distillation Configuration (Stage 2) ---
        # Loss = (1 - alpha) * CE + alpha * T^2 * KL
        self.distillation_alpha = 0.5
        self.distillation_temp = 4.0

        # --- Cross-Validation ---
        self.n_folds = 5

        # --- Caching ---
        # Path for caching processed datasets (e.g., resized images)
        self.cache_dir = os.path.join(self.working_dir, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    def to_dict(self):
        """Returns configuration as a dictionary for logging purposes."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("__")}
