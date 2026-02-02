import os
import torch


class Config:
    """
    Configuration for the Logic-Gated Residual-Hybrid Network (LGRH-Net).
    Centralizes hyperparameters, file paths, and architecture settings.
    """

    # ==========================================
    # General & Reproducibility
    # ==========================================
    seed = 42
    debug = False  # Set True to use subset of data for rapid debugging
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 4  # Number of dataloader workers

    # ==========================================
    # File Paths
    # ==========================================
    # Base directories
    input_dir = "./input"
    metadata_dir = "./metadata"
    working_dir = "./working"
    submission_dir = "./submission"

    # Cache directory for this specific idea (idea_27)
    cache_dir = os.path.join(working_dir, "idea_27")

    # Data files (using metadata splits)
    train_file = os.path.join(metadata_dir, "train.csv")
    val_file = os.path.join(metadata_dir, "validation.csv")
    test_file = os.path.join(metadata_dir, "test.csv")

    # Output files
    submission_file = os.path.join(submission_dir, "submission.csv")
    model_save_path = os.path.join(cache_dir, "best_model.pth")
    scaler_save_path = os.path.join(cache_dir, "scaler.joblib")

    # ==========================================
    # Model Architecture: LGRH-Net
    # ==========================================
    # Branch 1: Deep Residual Dense TCN (Resistive Stream)
    # No stem (direct input), large kernels, dense dilation
    tcn_kernel_size = 9
    tcn_filters = 256  # Increased capacity to match LSTM branch (Cite Lesson 57)
    tcn_layers = 8  # Deep stack of Residual Dense Blocks
    tcn_dropout = 0.1
    tcn_dilation = 1  # Strictly dense to preserve local fidelity

    # Branch 2: High-Capacity Bidirectional LSTM (Elastic Stream)
    lstm_hidden_size = 512
    lstm_layers = 3
    lstm_bidirectional = True
    lstm_dropout = 0.1

    # Fusion Head: Wide-Latent Integration
    fusion_hidden_size = 1024

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    epochs = 80
    batch_size = 128
    learning_rate = 1e-3
    weight_decay = 1e-4  # Low weight decay as per Lesson 93
    max_grad_norm = 1.0  # Gradient clipping as per Lesson 65
    patience = 15  # Early stopping patience

    # Learning Rate Scheduler
    scheduler_type = "CosineAnnealingWarmRestarts"
    T_0 = 10
    T_mult = 2
    eta_min = 1e-6

    # ==========================================
    # Feature Engineering
    # ==========================================
    # Lookahead window for u_in features
    lookahead_steps = 4

    # Caching behavior
    load_cached_data = True

    @classmethod
    def setup(cls):
        """Creates necessary directories if they don't exist."""
        os.makedirs(cls.cache_dir, exist_ok=True)
        os.makedirs(cls.submission_dir, exist_ok=True)

    def __init__(self, **kwargs):
        """
        Initialize config with optional overrides.
        """
        # Apply overrides
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

        # Ensure directories exist
        self.setup()

    def __repr__(self):
        """Pretty print configuration."""
        return "\n".join(
            [f"{k}: {v}" for k, v in self.__dict__.items() if not k.startswith("__")]
        )
