import os
import torch


class Config:
    """
    Configuration class for the Tweet Sentiment Extraction task.
    Centralizes hyperparameters, file paths, and model settings.
    """

    def __init__(self, debug=False, epochs=5, train_batch_size=8):
        # ====================================================
        # General Settings
        # ====================================================
        self.seed = 42
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_workers = 4
        self.debug = debug
        # If debug is True, use a small subset of data for rapid iteration
        self.debug_subset_size = 100

        # ====================================================
        # File Paths
        # ====================================================
        self.base_dir = "./"
        self.input_dir = os.path.join(self.base_dir, "input")
        self.metadata_dir = os.path.join(self.base_dir, "metadata")

        # Working directory for artifacts (models, cache)
        self.working_dir = os.path.join(self.base_dir, "working", "idea_10")

        # Data Paths (using generated metadata for stratification)
        self.train_path = os.path.join(self.metadata_dir, "train.csv")
        self.val_path = os.path.join(self.metadata_dir, "val.csv")
        self.test_path = os.path.join(self.metadata_dir, "test.csv")
        self.sample_submission_path = os.path.join(
            self.input_dir, "sample_submission.csv"
        )

        # Output Paths
        self.cache_dir = os.path.join(self.working_dir, "cache")
        self.model_dir = os.path.join(self.working_dir, "models")
        self.submission_dir = os.path.join(self.base_dir, "submission")
        self.submission_path = os.path.join(self.submission_dir, "submission.csv")

        # Ensure all necessary directories exist
        self._create_dirs()

        # ====================================================
        # Model Architecture
        # ====================================================
        self.model_name = "microsoft/deberta-v3-large"
        self.hidden_size = 1024  # Hidden size for DeBERTa-v3-large
        self.dropout = 0.1

        # Structural Innovation: CNN-Enhanced Span Head & Weighted Pooling
        self.num_pooling_layers = (
            4  # Number of last hidden states to use in weighted pooling
        )
        self.cnn_kernel_size = 3
        self.cnn_padding = 1
        self.cnn_mid_channels = 512  # Intermediate channel size for the CNN head

        # ====================================================
        # Tokenizer & Input
        # ====================================================
        self.max_len = 128  # Sufficient for tweet length + special tokens

        # ====================================================
        # Training Hyperparameters
        # ====================================================
        self.epochs = epochs
        self.train_batch_size = train_batch_size
        self.valid_batch_size = 16
        self.gradient_accumulation_steps = 1
        self.max_grad_norm = 1.0
        self.mixed_precision = True  # Enable AMP (Automatic Mixed Precision)

        # Optimizer & Scheduler
        self.learning_rate = 1e-5  # Uniform Learning Rate
        self.weight_decay = 0.01
        self.scheduler_type = "cosine"
        self.warmup_ratio = 0.1

        # Loss Function
        self.label_smoothing = 0.1

        # ====================================================
        # Adversarial Weight Perturbation (AWP)
        # ====================================================
        self.use_awp = True
        self.awp_start_epoch = 2  # Start AWP from this epoch
        self.awp_eps = 1e-2
        self.awp_lr = 1e-4

        # ====================================================
        # Cross-Validation
        # ====================================================
        self.n_folds = 5

    def _create_dirs(self):
        """Creates necessary directories for output and caching safely."""
        os.makedirs(self.working_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.submission_dir, exist_ok=True)

    def __repr__(self):
        return str(self.__dict__)
