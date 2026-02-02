import os
import torch


class Config:
    """
    Configuration class for the Pseudo-Labeled DeBERTa-v3-Large Ensemble task.
    Centralizes all hyperparameters, file paths, and model settings.
    """

    # ==========================================
    # General Settings
    # ==========================================
    seed = 42
    seeds = [42, 43, 44]  # Seeds for the ensemble (Teacher and Student models)
    debug = False  # Set to True to run on a small subset for debugging
    debug_subset_size = 100
    num_workers = 2  # Number of dataloader workers

    # ==========================================
    # Data Paths
    # ==========================================
    # Metadata paths (pre-split CSVs)
    train_path = "./metadata/train.csv"
    val_path = "./metadata/val.csv"
    test_path = "./metadata/test.csv"

    # Output and Artifact Paths
    working_dir = "./working/idea_6/"
    output_dir = os.path.join(working_dir, "models")
    cache_dir = os.path.join(working_dir, "cache")
    submission_path = "./submission/submission.csv"

    # ==========================================
    # Model Architecture
    # ==========================================
    model_name = "microsoft/deberta-v3-large"
    max_length = 160  # Max sequence length to preserve context
    dropout = 0.2  # Dropout rate for regularization
    freeze_layers = 6  # Number of bottom encoder layers (plus embeddings) to freeze

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    # Batch sizes
    train_batch_size = 4  # Physical batch size per step (fits in GPU memory)
    valid_batch_size = 16
    gradient_accumulation_steps = 4  # Effective batch size = 4 * 4 = 16

    # Optimization
    learning_rate = 1e-5
    weight_decay = 0.01
    max_grad_norm = 1.0

    # Scheduler
    num_epochs = 3
    scheduler_type = "linear"
    warmup_ratio = 0.1

    # Early Stopping
    patience = 2

    # ==========================================
    # Semi-Supervised Learning (Pseudo-Labeling)
    # ==========================================
    pseudo_label_threshold = 0.95  # Confidence threshold for assigning hard labels

    # ==========================================
    # Hardware
    # ==========================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def setup(cls):
        """
        Creates necessary directories for artifacts and caching.
        Should be called at the start of the pipeline.
        """
        os.makedirs(cls.working_dir, exist_ok=True)
        os.makedirs(cls.output_dir, exist_ok=True)
        os.makedirs(cls.cache_dir, exist_ok=True)
        os.makedirs(os.path.dirname(cls.submission_path), exist_ok=True)


# Automatically setup directories when config is imported
Config.setup()
