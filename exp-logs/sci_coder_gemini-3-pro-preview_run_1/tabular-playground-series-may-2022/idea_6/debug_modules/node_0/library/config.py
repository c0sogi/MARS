import os


class Config:
    """
    Global configuration for the Granular Unified Transformer (GUT) experiment.
    Defines hyperparameters, file paths, and model architecture settings.
    """

    def __init__(self, debug: bool = False):
        """
        Initialize configuration.

        Args:
            debug (bool): If True, enables debug mode with reduced epochs and dataset size
                          for rapid testing.
        """
        # -------------------------------------------------------------------
        # 1. General & Runtime Settings
        # -------------------------------------------------------------------
        self.debug = debug
        self.seed = 42
        self.device = "cuda"
        self.num_workers = 4  # Number of dataloader workers

        # -------------------------------------------------------------------
        # 2. Paths & Directories
        # -------------------------------------------------------------------
        self.input_dir = "./input"
        self.metadata_dir = "./metadata"

        # Output directory for checkpoints and cache
        self.working_dir = "./working/idea_6"
        os.makedirs(self.working_dir, exist_ok=True)

        # Metadata file paths (using pre-split stratified data)
        self.train_path = os.path.join(self.metadata_dir, "train.csv")
        self.val_path = os.path.join(self.metadata_dir, "val.csv")
        self.test_path = os.path.join(self.metadata_dir, "test.csv")

        # Submission
        self.submission_dir = "./submission"
        os.makedirs(self.submission_dir, exist_ok=True)
        self.submission_path = os.path.join(self.submission_dir, "submission.csv")

        # -------------------------------------------------------------------
        # 3. Data Configuration
        # -------------------------------------------------------------------
        self.id_col = "id"
        self.target_col = "target"
        self.sequence_col = "f_27"

        # Numerical Features: f_00 to f_30 (excluding f_27)
        # We also include the engineered feature 'unique_characters'
        self.numerical_features = [f"f_{i:02d}" for i in range(31) if i != 27]
        self.numerical_features.append("unique_characters")

        # Sequence Features (f_27)
        self.sequence_len = 10  # Fixed length of the string
        self.vocab_size = 40  # Sufficient for A-Z + padding + unknown

        # -------------------------------------------------------------------
        # 4. Model Hyperparameters (Granular Unified Transformer)
        # -------------------------------------------------------------------
        # Transformer Encoder
        self.d_model = 128
        self.n_heads = 4
        self.n_layers = 4
        self.dim_feedforward = 512
        self.dropout = 0.1

        # Head
        self.mlp_hidden_dims = [1024, 512, 256]

        # -------------------------------------------------------------------
        # 5. Training Hyperparameters
        # -------------------------------------------------------------------
        self.learning_rate = 1e-3
        self.weight_decay = 1e-2

        # Scheduler (OneCycleLR)
        self.pct_start = 0.3
        self.div_factor = 25.0
        self.final_div_factor = 1000.0

        # Early Stopping
        self.early_stopping_patience = 5
        self.early_stopping_min_delta = 1e-4

        # Dynamic settings based on debug flag
        if self.debug:
            self.epochs = 2
            self.batch_size = 64
            self.max_samples = 5000  # Limit dataset size for debugging
        else:
            self.epochs = 30
            self.batch_size = 2048  # Optimized for A100 GPU
            self.max_samples = None  # Use full dataset
