import os
import torch
import warnings


class Config:
    def __init__(self, debug: bool = False):
        """
        Configuration class for the Semi-Supervised DeBERTa-v3 pipeline.

        Args:
            debug (bool): If True, runs in debug mode with smaller dataset subsets
                          and fewer epochs to verify pipeline functionality.
        """
        # =================================================================
        # General Settings
        # =================================================================
        self.seed = 42
        self.debug = debug
        self.num_workers = 4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Suppress warnings
        warnings.filterwarnings("ignore")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # =================================================================
        # File Paths & Directories
        # =================================================================
        self.input_dir = "./input"
        self.metadata_dir = "./metadata"

        # Working directory for this specific idea iteration
        self.working_dir = "./working/idea_11"
        self.output_dir = os.path.join(self.working_dir, "output")
        self.cache_dir = self.working_dir  # For caching processed data

        # Create necessary directories
        os.makedirs(self.working_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        # Data Files (Metadata based)
        self.train_meta_file = os.path.join(self.metadata_dir, "train.csv")
        self.val_meta_file = os.path.join(self.metadata_dir, "val.csv")
        self.test_meta_file = os.path.join(self.metadata_dir, "test.csv")

        # Raw Data Files (for text merging)
        self.train_raw_file = os.path.join(self.input_dir, "train.csv")
        self.test_raw_file = os.path.join(self.input_dir, "test.csv")
        self.sample_submission_file = os.path.join(
            self.input_dir, "sample_submission.csv"
        )

        # Submission output
        self.submission_file = os.path.join(self.output_dir, "submission.csv")

        # =================================================================
        # Data Parameters
        # =================================================================
        self.target_cols = [
            "toxic",
            "severe_toxic",
            "obscene",
            "threat",
            "insult",
            "identity_hate",
        ]
        self.num_labels = len(self.target_cols)

        # Text Processing
        self.max_len = 256  # Sufficient for most comments, fits well in A100 memory
        self.truncation = True
        self.padding = "max_length"

        # Debugging constraints
        if self.debug:
            self.debug_sample_size = 1000  # Number of samples to use in debug mode
        else:
            self.debug_sample_size = None

        # =================================================================
        # Model Architecture
        # =================================================================
        self.model_name = "microsoft/deberta-v3-base"
        self.hidden_size = 768
        self.dropout = 0.1

        # Custom Head Components
        self.use_weighted_layer_aggregation = True
        self.aggregation_layers = 4  # Last N layers

        self.use_hybrid_pooling = True  # Mean + Max + Attention

        self.use_multi_sample_dropout = True
        self.msd_dropout_rates = [0.1, 0.2, 0.3, 0.4, 0.5]

        # =================================================================
        # Training Pipeline Parameters
        # =================================================================

        # --- Common Training Params ---
        self.train_batch_size = 8
        self.valid_batch_size = 16
        self.accumulation_steps = 4
        self.max_grad_norm = 10.0
        self.weight_decay = 0.01
        self.scheduler_type = "cosine"  # 'cosine' or 'linear'
        self.warmup_ratio = 0.1
        self.early_stopping_patience = 3

        # --- Stage 1: Domain-Adaptive Pre-training (DAPT) ---
        # MLM on Train + Test data
        self.dapt_epochs = 1 if self.debug else 3
        self.dapt_lr = 5e-5
        self.mlm_probability = 0.15
        self.dapt_model_path = os.path.join(self.working_dir, "dapt_backbone")

        # --- Stage 2: Teacher Training (Supervised) ---
        # 5-Fold CV on Labeled Train Data
        self.teacher_epochs = 2 if self.debug else 5
        self.teacher_lr = 2e-5
        self.teacher_folds = 2 if self.debug else 5

        # Adversarial Weight Perturbation (AWP)
        self.use_awp = True
        self.awp_lr = 1e-4
        self.awp_eps = 1e-2
        self.awp_start_epoch = 1  # Start AWP after 1st epoch

        # --- Stage 3: Student Training (Self-Training) ---
        # Train on Train + Soft-Labeled Test
        self.student_epochs = 2 if self.debug else 4
        self.student_lr = 2e-5
        self.student_folds = 2 if self.debug else 5
        self.soft_label_weight = 1.0  # Weight for soft labels in loss function

    def print_config(self):
        """Prints the current configuration."""
        print("=" * 40)
        print(f"CONFIG (Debug={self.debug})")
        print("=" * 40)
        print(f"Device: {self.device}")
        print(f"Model: {self.model_name}")
        print(f"Max Length: {self.max_len}")
        print(f"Batch Size: {self.train_batch_size}")
        print(f"DAPT Epochs: {self.dapt_epochs}")
        print(f"Teacher Epochs: {self.teacher_epochs}")
        print(f"Student Epochs: {self.student_epochs}")
        print(f"Output Dir: {self.output_dir}")
        print("=" * 40)
