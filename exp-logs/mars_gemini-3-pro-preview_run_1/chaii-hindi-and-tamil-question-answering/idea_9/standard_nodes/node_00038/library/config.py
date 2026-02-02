import os
import torch


class Config:
    """
    Configuration class for the Question Answering task.
    Encapsulates all hyperparameters, file paths, and model settings.
    """

    def __init__(self, debug=False):
        # General Settings
        self.debug = debug
        self.seed = 42
        self.n_seeds = 5
        self.seeds = [42 + i for i in range(self.n_seeds)]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_workers = 8  # Optimized for the available vCPUs

        # File Paths
        self.input_dir = "./input"
        self.metadata_dir = "./metadata"
        self.working_dir = "./working/idea_9"
        self.submission_dir = "./submission"

        # Input Files (using metadata split to prevent leakage)
        self.train_path = os.path.join(self.metadata_dir, "train.csv")
        self.val_path = os.path.join(self.metadata_dir, "val.csv")
        self.test_path = os.path.join(self.metadata_dir, "test.csv")

        # Output Files
        self.submission_path = os.path.join(self.submission_dir, "submission.csv")

        # Create necessary directories
        os.makedirs(self.working_dir, exist_ok=True)
        os.makedirs(self.submission_dir, exist_ok=True)

        # Model Architecture
        self.model_name = "xlm-roberta-large"
        self.max_len = 384  # Maximum sequence length for the model
        self.doc_stride = 128  # Overlap between sliding windows

        # Training Hyperparameters
        self.epochs = 7
        self.batch_size = 8  # Small batch size for regularization/gradient noise
        self.lr_backbone = 1e-5  # Lower learning rate for the pre-trained backbone
        self.lr_head = 5e-5  # Higher learning rate for the task-specific heads
        self.weight_decay = 0.01  # Global weight decay applied to ALL parameters
        self.max_grad_norm = 1.0
        self.use_amp = True  # Automatic Mixed Precision

        # Loss Configuration
        self.aux_weight = 0.5  # Weight for the relevance (binary classification) loss

        # Adversarial Training (FGM)
        self.use_fgm = False
        self.fgm_epsilon = 1.0
        self.fgm_name = "word_embeddings"  # Target parameter name for perturbation

        # Data Processing Strategy
        self.negative_positive_ratio = (
            2  # Ratio of negative (no answer) to positive windows
        )
        self.use_full_data = True  # Concatenate train and val sets for final training

        # Debugging / Development
        self.debug_sample_size = 50  # Number of samples to use when debug=True

    def __str__(self):
        """Returns a string representation of the configuration."""
        config_dict = {k: v for k, v in self.__dict__.items() if not k.startswith("__")}
        return str(config_dict)
