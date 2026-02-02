import os
import torch


class Config:
    """
    Configuration class for the Question Answering task using XLM-Roberta-Large.
    Centralizes paths, hyperparameters, and model settings.
    """

    def __init__(self, debug: bool = False, epochs: int = 10, batch_size: int = 4):
        """
        Initialize configuration with flexible overrides.

        Args:
            debug (bool): If True, runs in debug mode with fewer data/epochs.
            epochs (int): Number of training epochs per seed.
            batch_size (int): Training batch size.
        """
        # =====================================================================
        # PATHS
        # =====================================================================
        self.input_dir = "./input"
        self.metadata_dir = "./metadata"

        # Working directory for Idea 12
        self.working_dir = "./working/idea_12"
        self.cache_dir = os.path.join(self.working_dir, "cache")
        self.output_dir = os.path.join(self.working_dir, "output")
        self.submission_dir = "./submission"

        # Ensure directories exist
        os.makedirs(self.working_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.submission_dir, exist_ok=True)

        # =====================================================================
        # MODEL ARCHITECTURE
        # =====================================================================
        self.model_name = "xlm-roberta-large"
        self.hidden_dropout = 0.1
        self.attention_dropout = 0.1

        # =====================================================================
        # DATA PROCESSING
        # =====================================================================
        self.max_len = 384
        self.doc_stride = 128

        # Negative Sampling: 2:1 Negative-to-Positive ratio
        # We retain 100% of positives and downsample negatives.
        self.neg_pos_ratio = 2.0

        # =====================================================================
        # TRAINING STRATEGY
        # =====================================================================
        self.debug = debug

        # Ensemble Strategy: 5 independent seeds (Cite Lesson 46)
        self.seeds = [42, 43, 44, 45, 46]

        self.epochs = epochs
        self.train_batch_size = batch_size
        self.valid_batch_size = 32  # Larger batch size for inference/validation

        # Hardware
        self.num_workers = 2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # =====================================================================
        # OPTIMIZATION
        # =====================================================================
        # Differential Learning Rates
        self.lr_backbone = 1e-5
        self.lr_heads = 5e-5

        # Global Weight Decay (applied to all params including bias/LayerNorm)
        self.weight_decay = 0.01

        self.max_grad_norm = 1.0

        # Loss Weighting: L_total = L_span + 0.5 * L_relevance
        self.relevance_loss_weight = 0.5

        # =====================================================================
        # ADVERSARIAL TRAINING (FGM)
        # =====================================================================
        self.use_fgm = True
        self.fgm_epsilon = 1.0

        # =====================================================================
        # DEBUG OVERRIDES
        # =====================================================================
        if self.debug:
            self.seeds = [42]  # Run single seed for debugging
            self.epochs = 2
            print(f"Debug mode active: Epochs={self.epochs}, Seeds={self.seeds}")
