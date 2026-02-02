import os
import torch


class ModelConfig:
    """
    Configuration for the Cross-Validated DeBERTa-v3-Large Ensemble.
    """

    def __init__(self):
        # ==========================================
        # General Settings
        # ==========================================
        self.seed = 42
        self.debug = False  # Set to True to run on a small subset for debugging

        # ==========================================
        # File Paths
        # ==========================================
        self.input_dir = "./metadata"
        self.train_path = os.path.join(self.input_dir, "train.csv")
        # Note: val.csv exists but we use n-fold CV on train.csv
        self.val_path = os.path.join(self.input_dir, "val.csv")
        self.test_path = os.path.join(self.input_dir, "test.csv")

        # Directory for caching intermediate artifacts (Parquet/Numpy)
        self.working_dir = "./working/idea_5"
        os.makedirs(self.working_dir, exist_ok=True)

        # Directory for final submission
        self.output_dir = "./submission"
        os.makedirs(self.output_dir, exist_ok=True)

        # ==========================================
        # Model Architecture
        # ==========================================
        self.model_name = "microsoft/deberta-v3-large"
        self.max_len = 160
        self.dropout = 0.2
        self.freeze_layers = 6  # Freeze embeddings and bottom 6 encoder layers

        # ==========================================
        # Training Hyperparameters
        # ==========================================
        self.n_folds = 5
        self.epochs = 4
        self.learning_rate = 1e-5
        self.weight_decay = 0.01
        self.max_grad_norm = 1.0

        # ==========================================
        # Batch Size & Gradient Accumulation
        # ==========================================
        # Physical batch size (fits in GPU memory)
        self.train_batch_size = 8
        self.valid_batch_size = 16

        # Target effective batch size for stable optimization
        self.target_effective_batch_size = 32

        # Calculate accumulation steps
        self.accumulation_steps = max(
            1, self.target_effective_batch_size // self.train_batch_size
        )

        # ==========================================
        # Scheduler & Optimization
        # ==========================================
        self.scheduler_type = "linear"
        self.warmup_ratio = 0.1
        self.patience = 2  # Early stopping patience

        # ==========================================
        # Hardware
        # ==========================================
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_workers = 4
        self.pin_memory = True

    def display(self):
        """Prints the configuration."""
        print("=" * 30)
        print("Model Configuration")
        print("=" * 30)
        for k, v in self.__dict__.items():
            print(f"{k}: {v}")
        print("=" * 30)
