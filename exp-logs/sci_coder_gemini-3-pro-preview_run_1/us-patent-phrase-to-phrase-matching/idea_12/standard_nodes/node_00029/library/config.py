import os
import torch


class Config:
    """
    Configuration class for the Phrase Similarity scoring task.
    Centralizes all hyperparameters, paths, and model settings.
    """

    def __init__(self, debug: bool = False, epochs: int = 4):
        # =================================================================
        # General Settings
        # =================================================================
        self.seed = 42
        self.debug = debug
        self.debug_sample_size = 200  # Number of samples to use in debug mode

        # =================================================================
        # Paths
        # =================================================================
        self.input_root = "./metadata"
        # Using metadata splits as requested
        self.train_path = os.path.join(self.input_root, "train.csv")
        self.val_path = os.path.join(self.input_root, "val.csv")
        self.test_path = os.path.join(self.input_root, "test.csv")
        self.sample_submission_path = "./input/sample_submission.csv"

        # Output directory for checkpoints, cache, and predictions
        self.output_dir = "./working/idea_12"
        os.makedirs(self.output_dir, exist_ok=True)

        # =================================================================
        # Model Architecture
        # =================================================================
        self.model_name = "microsoft/deberta-v3-large"
        # Max length set to 140 to safely accommodate:
        # [CLS] [Context_Token] Anchor [SEP] Target [SEP]
        self.max_length = 140
        self.num_classes = 1  # Regression output

        # =================================================================
        # Training Hyperparameters
        # =================================================================
        self.epochs = epochs

        # Batch size tailored for A100 40GB with DeBERTa-Large
        self.train_batch_size = 8
        self.valid_batch_size = 16
        self.gradient_accumulation_steps = 1

        # Optimizer settings
        self.learning_rate = 2e-5
        self.weight_decay = 0.01
        self.eps = 1e-6
        self.max_grad_norm = 1.0

        # Scheduler settings
        self.warmup_ratio = 0.1
        self.scheduler_type = "cosine"

        # Layer-wise Learning Rate Decay (LLRD)
        self.llrd_decay = 0.9

        # =================================================================
        # Cross-Validation Strategy
        # =================================================================
        self.n_folds = 5
        self.target_col = "score"

        # =================================================================
        # Hardware / Environment
        # =================================================================
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_workers = 4
        self.pin_memory = True

        # =================================================================
        # Debug Overrides
        # =================================================================
        if self.debug:
            print(
                f"DEBUG MODE ENABLED: Reducing data size to {self.debug_sample_size} and epochs to 2."
            )
            self.epochs = 2
            self.n_folds = 2
            self.train_batch_size = 4
            self.valid_batch_size = 4
