import os
import torch


class Config:
    def __init__(self):
        # =============================================================================
        # General Configuration
        # =============================================================================
        self.seed = 42
        self.debug = (
            False  # Set to True to run with a small subset of data for debugging
        )
        self.num_workers = 4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.print_freq = 50

        # =============================================================================
        # Directories & Paths
        # =============================================================================
        self.input_dir = "./input"
        self.metadata_dir = "./metadata"
        self.working_dir = "./working/idea_6"

        # Ensure working directory exists
        os.makedirs(self.working_dir, exist_ok=True)

        # File Paths for Caching and Outputs
        self.context_map_path = os.path.join(self.working_dir, "context_map.parquet")
        self.train_cache_path = os.path.join(self.working_dir, "train_cache.parquet")
        self.val_cache_path = os.path.join(self.working_dir, "val_cache.parquet")
        self.test_cache_path = os.path.join(self.working_dir, "test_cache.parquet")

        self.model_output_path = os.path.join(self.working_dir, "best_model.pth")
        self.submission_path = os.path.join(self.working_dir, "submission.csv")

        # DAPT Model Path (Pre-trained on domain data)
        self.dapt_model_dir = os.path.join(self.working_dir, "dapt_model")

        # =============================================================================
        # Model Architecture
        # =============================================================================
        self.model_name = "microsoft/deberta-v3-large"
        self.max_len = 130  # Max sequence length (Context Hierarchy + Anchor + Target)
        self.dropout = 0.2
        self.num_classes = (
            5  # For auxiliary classification head (0, 0.25, 0.5, 0.75, 1.0)
        )

        # =============================================================================
        # Training Hyperparameters
        # =============================================================================
        self.epochs = 5
        self.batch_size = 8  # Adjusted for A100 (40GB) with Large model
        self.gradient_accumulation_steps = 1
        self.max_grad_norm = 1000

        # Optimizer
        self.learning_rate = 2e-5
        self.weight_decay = 0.01
        self.eps = 1e-6
        self.betas = (0.9, 0.999)

        # Scheduler
        self.scheduler_type = "cosine"
        self.warmup_ratio = 0.1
        self.batch_scheduler = True

        # Layer-wise Learning Rate Decay (LLRD)
        self.llrd_decay = 0.9

        # =============================================================================
        # Advanced Training Strategies
        # =============================================================================
        # Adversarial Weight Perturbation (AWP)
        self.use_awp = True
        self.awp_eps = 1e-4
        self.awp_lr = 1e-4
        self.awp_start_epoch = 1.0  # Start AWP after this many epochs

        # Loss Function Weights
        # Total Loss = mse_weight * MSE + ce_weight * CE + pearson_weight * (1 - Pearson)
        self.mse_weight = 1.0
        self.ce_weight = 0.5
        self.pearson_weight = 1.0

        # =============================================================================
        # Cross-Validation
        # =============================================================================
        self.n_fold = 5
        self.trn_folds = [0]  # Folds to train in this run (can be multiple)


# Instantiate the config to be imported by other modules
cfg = Config()
