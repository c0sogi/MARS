import os
import torch


class Config:
    def __init__(self):
        # -------------------------------------------------------------------
        # Paths and Directories
        # -------------------------------------------------------------------
        self.input_dir = "./input"
        self.metadata_dir = "./metadata"
        self.working_dir = "./working/idea_8"

        # Sub-directories for organization
        self.cache_dir = os.path.join(self.working_dir, "qa_cache")
        self.model_dir = os.path.join(self.working_dir, "qa_models")
        self.submission_dir = os.path.join(self.working_dir, "submission")

        # TAPT specific paths
        self.tapt_cache_dir = os.path.join(self.working_dir, "tapt_cache")
        self.tapt_output_dir = os.path.join(self.working_dir, "tapt_model_finetuned")

        # Create directories
        os.makedirs(self.working_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.submission_dir, exist_ok=True)
        os.makedirs(self.tapt_cache_dir, exist_ok=True)
        os.makedirs(self.tapt_output_dir, exist_ok=True)

        # -------------------------------------------------------------------
        # Model Configuration
        # -------------------------------------------------------------------
        self.model_name = "xlm-roberta-base"
        self.num_labels = 3  # O, B-ANS, I-ANS

        # -------------------------------------------------------------------
        # Data Preprocessing / Sliding Window
        # -------------------------------------------------------------------
        self.max_length = 384
        self.doc_stride = 128

        # -------------------------------------------------------------------
        # Training Hyperparameters
        # -------------------------------------------------------------------
        self.seed = 42
        self.seeds = [42, 43, 44]  # For ensembling

        self.batch_size = 16
        self.epochs = 10
        self.learning_rate = 2e-5
        self.weight_decay = 0.01
        self.warmup_ratio = 0.1

        # Class Weights for Loss Function (O, B-ANS, I-ANS)
        # B and I are weighted significantly higher to handle class imbalance
        self.class_weights = torch.tensor([1.0, 10.0, 10.0])

        # -------------------------------------------------------------------
        # Inference / Post-processing
        # -------------------------------------------------------------------
        self.n_best_size = 20
        self.max_answer_length = 30

        # -------------------------------------------------------------------
        # Hardware
        # -------------------------------------------------------------------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_workers = 4

    def get_tapt_config(self):
        """Returns specific config for TAPT phase."""
        return {
            "model_name": self.model_name,
            "output_dir": self.tapt_output_dir,
            "train_batch_size": 8,  # Slightly smaller for MLM if needed, or same
            "num_train_epochs": 3,  # TAPT usually requires fewer epochs
            "learning_rate": 2e-5,
            "mlm_probability": 0.15,
        }
