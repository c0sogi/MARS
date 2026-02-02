import os
import torch


class Config:
    """
    Configuration class for the Toxic Comment Classification task.
    Implements settings for Context-Aware DeBERTa-v3-Base with AWP.
    """

    def __init__(self, debug=False, epochs=5, batch_size=16):
        """
        Initialize configuration with optional overrides.

        Args:
            debug (bool): If True, limits dataset size for debugging.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
        """
        # ==========================================
        # General Settings
        # ==========================================
        self.seed = 42
        self.debug = debug

        # ==========================================
        # Directory & File Paths
        # ==========================================
        self.input_dir = "./input"
        self.metadata_dir = "./metadata"
        self.working_dir = "./working/idea_7"

        # Ensure working directory exists
        os.makedirs(self.working_dir, exist_ok=True)

        # Metadata Paths (Stratified Splits)
        self.train_meta_path = os.path.join(self.metadata_dir, "train.csv")
        self.val_meta_path = os.path.join(self.metadata_dir, "val.csv")
        self.test_meta_path = os.path.join(self.metadata_dir, "test.csv")

        # Raw Data Paths (Text Content)
        self.train_data_path = os.path.join(self.input_dir, "train.csv")
        self.test_data_path = os.path.join(self.input_dir, "test.csv")
        self.sample_submission_path = os.path.join(
            self.input_dir, "sample_submission.csv"
        )

        # Output Paths
        self.model_save_path = os.path.join(self.working_dir, "model.pth")
        self.submission_path = os.path.join(self.working_dir, "submission.csv")

        # ==========================================
        # Data Configuration
        # ==========================================
        self.target_cols = [
            "toxic",
            "severe_toxic",
            "obscene",
            "threat",
            "insult",
            "identity_hate",
        ]
        self.num_classes = len(self.target_cols)
        self.num_workers = 4

        # ==========================================
        # Model Architecture
        # ==========================================
        self.model_name = "microsoft/deberta-v3-base"
        self.max_len = 300
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1

        # ==========================================
        # Training Hyperparameters
        # ==========================================
        self.epochs = epochs
        self.train_batch_size = batch_size
        self.valid_batch_size = batch_size * 2

        # Optimizer (AdamW)
        self.lr = 2e-5
        self.min_lr = 1e-6
        self.weight_decay = 0.01
        self.max_grad_norm = 1.0

        # Scheduler (OneCycleLR)
        self.scheduler_type = "OneCycleLR"
        self.pct_start = 0.1  # Warmup percentage

        # ==========================================
        # Adversarial Weight Perturbation (AWP)
        # ==========================================
        self.use_awp = True
        self.awp_lr = 1e-4
        self.awp_eps = 1e-2
        # Start AWP after the first epoch (0-indexed, so start at epoch 1)
        # This allows the model to stabilize before adversarial attacks
        self.awp_start_epoch = 1

        # ==========================================
        # Hardware
        # ==========================================
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __str__(self):
        """Returns a string representation of the configuration."""
        config_dict = {k: v for k, v in self.__dict__.items() if not k.startswith("__")}
        return str(config_dict)
