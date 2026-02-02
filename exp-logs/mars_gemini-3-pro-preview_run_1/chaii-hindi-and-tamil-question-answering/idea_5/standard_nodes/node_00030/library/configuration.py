import os
import torch


class Config:
    """
    Configuration class for the Question Answering task.
    Centralizes all hyperparameters, file paths, and model settings.
    """

    def __init__(self):
        # =============================================================================
        # PATHS & DIRECTORIES
        # =============================================================================
        self.input_dir = "./input"
        self.metadata_dir = "./metadata"
        self.working_dir = "./working/idea_5"
        self.submission_dir = "./submission"

        # Input Data Files (Generated Metadata)
        self.train_meta_path = os.path.join(self.metadata_dir, "train.csv")
        self.val_meta_path = os.path.join(self.metadata_dir, "val.csv")
        self.test_meta_path = os.path.join(self.metadata_dir, "test.csv")
        self.sample_submission_path = os.path.join(
            self.input_dir, "sample_submission.csv"
        )

        # Output/Cache Files
        # We will cache the processed features to avoid re-tokenizing every run
        self.train_features_path = os.path.join(
            self.working_dir, "train_features.parquet"
        )
        self.test_features_path = os.path.join(
            self.working_dir, "test_features.parquet"
        )

        # Model Checkpoint & Submission
        self.best_model_path = os.path.join(self.working_dir, "best_model.pth")
        self.submission_path = os.path.join(self.submission_dir, "submission.csv")

        # =============================================================================
        # MODEL ARCHITECTURE & TOKENIZATION
        # =============================================================================
        # Using the Large variant as per strategy for better cross-lingual transfer
        self.model_name = "xlm-roberta-large"

        # Sliding Window Configuration
        self.max_length = 384  # Maximum sequence length for the model
        self.doc_stride = 128  # Overlap between windows

        # =============================================================================
        # TRAINING HYPERPARAMETERS
        # =============================================================================
        self.seed = 42
        self.batch_size = (
            4  # Small batch size for Large model stability & regularization
        )
        self.epochs = 5  # Fixed training duration
        self.learning_rate = 1e-5  # Lower learning rate for fine-tuning Large model
        self.weight_decay = 0.01
        self.warmup_ratio = 0.1
        self.max_grad_norm = 1.0

        # =============================================================================
        # DATA PROCESSING & SAMPLING STRATEGY
        # =============================================================================
        # Negative Sampling: Ratio of negative (no-answer) windows to positive windows.
        # Strategy: Keep all positives, sample negatives to maintain 2:1 ratio.
        self.negative_sampling_ratio = 2.0

        # =============================================================================
        # MULTI-TASK LOSS CONFIGURATION
        # =============================================================================
        # Loss = Span_Loss + (lambda * Relevance_Loss)
        # Relevance head helps filter out irrelevant windows during inference.
        self.relevance_loss_weight = 0.5

        # =============================================================================
        # INFERENCE & POST-PROCESSING
        # =============================================================================
        self.n_best_size = 20  # Number of candidate spans to consider
        self.max_answer_length = 30  # Maximum length of a predicted answer

        # =============================================================================
        # HARDWARE
        # =============================================================================
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_workers = 2

        # Initialize directories
        self._create_directories()

    def _create_directories(self):
        """Creates necessary working directories if they don't exist."""
        os.makedirs(self.working_dir, exist_ok=True)
        os.makedirs(self.submission_dir, exist_ok=True)
