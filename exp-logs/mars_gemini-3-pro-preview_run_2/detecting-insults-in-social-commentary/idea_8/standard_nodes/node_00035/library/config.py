import os
import torch


class Config:
    """
    Central configuration for the Insult Detection task.
    Implements settings for TAPT (Task-Adaptive Pre-Training) and
    Adversarial Fine-Tuning using DeBERTa-v3-Large.
    """

    def __init__(self, debug=False, epochs=5, train_batch_size=8):
        # General Settings
        self.debug = debug
        self.seed = 42
        self.num_workers = 4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # File Paths
        self.input_dir = "./input"
        self.metadata_dir = "./metadata"
        self.train_path = os.path.join(self.metadata_dir, "train.csv")
        self.val_path = os.path.join(self.metadata_dir, "val.csv")
        self.test_path = os.path.join(self.metadata_dir, "test.csv")
        self.sample_submission_path = os.path.join(
            self.input_dir, "sample_submission_null.csv"
        )

        # Output Directories
        self.output_dir = "./working/idea_8"
        self.submission_path = "./submission/submission.csv"
        self.tapt_output_dir = os.path.join(self.output_dir, "tapt_output")

        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.submission_path), exist_ok=True)
        os.makedirs(self.tapt_output_dir, exist_ok=True)

        # Data Column Definitions
        self.target_col = "Insult"
        self.text_col = "Comment"
        self.date_col = "Date"

        # Model Architecture
        self.model_name = "microsoft/deberta-v3-large"
        self.max_length = 256  # Balanced for A100 memory and text length distribution
        self.gradient_checkpointing = True  # Essential for Large models on 40GB VRAM
        self.pooler_type = "mean"  # Mean pooling of last hidden state

        # Multi-Sample Dropout (Head)
        self.use_msd = True
        self.msd_num = 5
        self.fc_dropout = 0.2

        # Training Hyperparameters
        self.epochs = epochs
        self.train_batch_size = train_batch_size
        self.valid_batch_size = 16
        self.learning_rate = 1e-5
        self.min_lr = 1e-6
        self.weight_decay = 0.01
        self.scheduler_type = "cosine"
        self.warmup_ratio = 0.1
        self.clip_grad_norm = 1.0

        # Layer-wise Learning Rate Decay (LLRD)
        self.use_llrd = True
        self.llrd_decay = 0.9

        # Adversarial Weight Perturbation (AWP)
        self.use_awp = True
        self.awp_lr = 1e-4
        self.awp_eps = 1e-2
        self.awp_start_epoch = 1  # Start AWP after the first epoch (warmup)

        # Task-Adaptive Pre-Training (TAPT)
        self.use_tapt = True
        self.tapt_epochs = 3
        self.tapt_batch_size = 8
        self.tapt_lr = 2e-5
        self.mlm_probability = 0.15

        # Cross-Validation
        self.num_folds = 5

        # Debugging
        if self.debug:
            self.epochs = 2
            self.tapt_epochs = 1
            self.train_batch_size = 4
            self.valid_batch_size = 4
