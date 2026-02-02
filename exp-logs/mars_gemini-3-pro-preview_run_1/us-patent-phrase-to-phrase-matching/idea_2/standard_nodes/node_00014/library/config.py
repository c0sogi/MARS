import os
import torch


class Config:
    """
    Configuration class for the Patent Phrase Similarity task.
    Centralizes all hyperparameters, file paths, and model settings.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    seed = 42
    debug = False  # Set to True to run on a small subset for debugging
    debug_sample_size = 1000  # Number of samples to use when debug=True

    # =========================================================================
    # File Paths
    # =========================================================================
    # Input data (using metadata splits)
    train_path = "./metadata/train.csv"
    val_path = "./metadata/val.csv"
    test_path = "./metadata/test.csv"
    sample_submission_path = "./input/sample_submission.csv"

    # Output directories
    output_dir = "./working/idea_2"
    model_output_dir = os.path.join(output_dir, "models")
    prediction_output_dir = os.path.join(output_dir, "predictions")
    cache_dir = os.path.join(output_dir, "cache")

    # Ensure directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(prediction_output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    # =========================================================================
    # Model Configuration
    # =========================================================================
    model_name = "microsoft/deberta-v3-large"
    num_classes = 1  # Regression task (Pearson correlation target)
    max_length = 140  # Sufficient for Context + Anchor + Target
    dropout = 0.0  # Dropout for the regression head

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    n_folds = 4  # Number of folds for Stratified Group K-Fold
    epochs = 10  # Maximum number of epochs (used with early stopping)

    # Batch sizes for A100 40GB
    train_batch_size = 16
    valid_batch_size = 32

    # Optimization
    learning_rate = 2e-5
    weight_decay = 0.01
    gradient_accumulation_steps = 1
    max_grad_norm = 1000.0

    # Scheduler
    scheduler_type = "linear"
    warmup_ratio = 0.1

    # Early Stopping
    patience = 3  # Stop if validation score doesn't improve for 3 epochs

    # =========================================================================
    # Hardware & Computation
    # =========================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 4
    pin_memory = True

    # =========================================================================
    # Utility Methods
    # =========================================================================
    @classmethod
    def print_config(cls):
        """Prints the current configuration."""
        print("=" * 30)
        print("CONFIG")
        print("=" * 30)
        for k, v in cls.__dict__.items():
            if not k.startswith("__") and not callable(v):
                print(f"{k}: {v}")
        print("=" * 30)
