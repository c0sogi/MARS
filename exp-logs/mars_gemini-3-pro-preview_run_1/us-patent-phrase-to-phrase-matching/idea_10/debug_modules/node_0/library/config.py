import os
import torch


class Config:
    """
    Configuration class for the DeBERTa-v3-large semantic similarity pipeline.
    Centralizes hyperparameters, file paths, and architectural settings.
    """

    def __init__(self, debug: bool = False):
        # =============================================================================
        # General Settings
        # =============================================================================
        self.debug = debug
        self.seed = 42
        self.num_workers = 4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # =============================================================================
        # File Paths
        # =============================================================================
        # Input Directories
        self.input_dir = "./input"
        self.metadata_dir = "./metadata"

        # Data Paths (using generated metadata)
        self.train_path = os.path.join(self.metadata_dir, "train.csv")
        self.val_path = os.path.join(self.metadata_dir, "val.csv")
        self.test_path = os.path.join(self.metadata_dir, "test.csv")
        self.cpc_path = os.path.join(self.input_dir, "description.md")
        self.sample_submission_path = os.path.join(
            self.input_dir, "sample_submission.csv"
        )

        # Output Directories
        self.working_dir = "./working/idea_10"
        self.output_dir = self.working_dir  # Alias for compatibility
        self.model_dir = os.path.join(self.working_dir, "models")
        self.predictions_dir = os.path.join(self.working_dir, "predictions")
        self.submission_dir = "./submission"
        self.submission_path = os.path.join(self.submission_dir, "submission.csv")

        # Ensure directories exist
        os.makedirs(self.working_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.predictions_dir, exist_ok=True)
        os.makedirs(self.submission_dir, exist_ok=True)

        # =============================================================================
        # Model Architecture
        # =============================================================================
        self.model_name = "microsoft/deberta-v3-large"
        # Max length sufficient for: [CLS] Context [SEP] Anchor [SEP] Target [SEP]
        self.max_length = 133
        self.gradient_checkpointing = True

        # Multi-Sample Dropout (MSD) Settings
        self.use_msd = True
        self.msd_samples = 5
        self.msd_dropout = 0.1

        # =============================================================================
        # Training Hyperparameters
        # =============================================================================
        self.n_folds = 5
        self.epochs = 4

        # Batch Size & Accumulation
        # A100 40GB allows decent batch size for Large models with FP16
        self.train_batch_size = 8
        self.valid_batch_size = 16
        self.gradient_accumulation_steps = 1
        self.max_grad_norm = 10.0

        # Optimizer (AdamW)
        self.learning_rate = 2e-5
        self.weight_decay = 0.01
        self.eps = 1e-6
        self.betas = (0.9, 0.999)

        # Scheduler (Cosine with Warmup)
        self.scheduler_type = "cosine"
        self.warmup_ratio = 0.1
        self.num_cycles = 0.5

        # Layer-wise Learning Rate Decay (LLRD)
        self.use_llrd = True
        self.llrd_decay = 0.9
        self.head_lr = 1e-4  # Higher learning rate for the newly initialized head

        # Loss Function
        self.loss_fn = "MSE"  # Mean Squared Error

        # =============================================================================
        # Debug / Quick Run Overrides
        # =============================================================================
        if self.debug:
            self.epochs = 2
            self.n_folds = 2
            self.train_batch_size = 4
            self.valid_batch_size = 4
            self.debug_sample_size = 100  # Only use a small subset of data
