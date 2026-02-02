import os
import random
import numpy as np
import torch


class Config:
    """
    Configuration class for DeBERTa-v3-Large with AWP and Hybrid Ranking.
    """

    def __init__(self, debug=False, epochs=4, train_batch_size=8):
        # ==========================================
        # General Settings
        # ==========================================
        self.seed = 42
        self.debug = debug
        # If debug is True, limit dataset size for rapid iteration
        self.train_subset_size = 5000 if self.debug else None

        # ==========================================
        # Paths
        # ==========================================
        # Input metadata paths (pre-generated)
        self.train_path = "./metadata/train.csv"
        self.val_path = "./metadata/val.csv"
        self.test_path = "./metadata/test.csv"
        self.sample_submission_path = "./input/sample_submission.csv"

        # Output directory for checkpoints and cache
        self.output_dir = "./working/idea_8"

        # ==========================================
        # Model Architecture
        # ==========================================
        self.model_name = "microsoft/deberta-v3-large"
        self.max_len = 320  # Sufficient context while managing memory for Large model

        # Target Configuration
        self.target_col = "target"
        self.aux_identity_cols = [
            "male",
            "female",
            "homosexual_gay_or_lesbian",
            "christian",
            "jewish",
            "muslim",
            "black",
            "white",
            "psychiatric_or_mental_illness",
        ]
        self.aux_attack_col = "identity_attack"

        # ==========================================
        # Training Hyperparameters
        # ==========================================
        self.epochs = epochs
        self.train_batch_size = train_batch_size
        self.valid_batch_size = 16
        self.learning_rate = 1e-5
        self.min_lr = 1e-6
        self.weight_decay = 0.01
        self.scheduler = "cosine"
        self.patience = 2  # For early stopping
        self.num_workers = 4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ==========================================
        # Loss Function Weights
        # ==========================================
        # Composite Loss = BCE + lambda_rank * RankLoss + lambda_aux * AuxLoss
        self.lambda_rank = 1.0
        self.lambda_aux = 0.5

        # ==========================================
        # Adversarial Weight Perturbation (AWP)
        # ==========================================
        self.use_awp = True
        self.awp_start_epoch = (
            2  # Start AWP after the model has stabilized (e.g., epoch 2)
        )
        self.awp_eps = 1e-2  # Perturbation size
        self.awp_lr = 1e-4  # Learning rate for the adversary

        # ==========================================
        # Environment Setup
        # ==========================================
        self.setup_environment()

    def setup_environment(self):
        """
        Creates necessary directories and sets random seeds for reproducibility.
        """
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Set random seeds
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        # Enforce deterministic behavior for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Suppress tokenizer parallelism warnings
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
