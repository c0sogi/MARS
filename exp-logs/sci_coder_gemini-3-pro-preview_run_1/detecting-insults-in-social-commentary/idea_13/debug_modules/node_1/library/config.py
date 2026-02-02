import os
import torch
import random
import numpy as np


class Config:
    # =========================================================================
    # General Configuration
    # =========================================================================
    seed = 42
    debug = False
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================================================================
    # File Paths
    # =========================================================================
    # Input data (Metadata)
    METADATA_DIR = "./metadata"
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "validation.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Working Directories
    WORKING_DIR = "./working/idea_13"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    MODEL_DIR = os.path.join(WORKING_DIR, "models")
    OUTPUT_DIR = os.path.join(WORKING_DIR, "outputs")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    for d in [WORKING_DIR, CACHE_DIR, MODEL_DIR, OUTPUT_DIR, SUBMISSION_DIR]:
        os.makedirs(d, exist_ok=True)

    # =========================================================================
    # Model Architectures
    # =========================================================================
    # Backbone models
    model_deberta = "microsoft/deberta-v3-large"
    model_roberta = "roberta-large"

    # List of models to iterate over
    models = [model_deberta, model_roberta]

    # =========================================================================
    # Data Processing
    # =========================================================================
    max_len = 256

    # Structural Features (SVD)
    svd_components = 256
    # N-gram ranges for feature extraction (Word 1-2, Char 3-5)
    tfidf_word_ngram_range = (1, 2)
    tfidf_char_ngram_range = (3, 5)

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    n_folds = 5
    trn_folds = [0, 1, 2, 3, 4]

    epochs = 4
    batch_size = 8
    grad_accum_steps = 2

    # Differential Learning Rates
    lr_backbone = 1e-5
    lr_head = 1e-3
    weight_decay = 0.01
    max_grad_norm = 1000

    # Scheduler
    scheduler_type = "cosine"
    warmup_ratio = 0.1
    num_cycles = 0.5

    # =========================================================================
    # Advanced Techniques
    # =========================================================================
    # Adversarial Weight Perturbation (AWP)
    use_awp = True
    awp_start_epoch = 2
    awp_lr = 1e-4
    awp_eps = 1e-4

    # Variable-Rate Multi-Sample Dropout (VR-MSD)
    dropout_rates = [0.1, 0.2, 0.3, 0.4, 0.5]

    # Stacking Meta-Learner
    meta_alpha = 1.0  # Ridge Regression Alpha

    # =========================================================================
    # Utility Methods
    # =========================================================================
    @classmethod
    def set_seed(cls):
        """Sets the random seed for reproducibility."""
        random.seed(cls.seed)
        np.random.seed(cls.seed)
        torch.manual_seed(cls.seed)
        torch.cuda.manual_seed(cls.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @classmethod
    def setup(cls, debug=False):
        """Configures the run based on debug mode."""
        cls.debug = debug
        if cls.debug:
            cls.epochs = 2
            cls.trn_folds = [0, 1]
            cls.batch_size = 4
            print(
                f"[Config] Debug mode enabled. Epochs: {cls.epochs}, Folds: {cls.trn_folds}"
            )
        cls.set_seed()
