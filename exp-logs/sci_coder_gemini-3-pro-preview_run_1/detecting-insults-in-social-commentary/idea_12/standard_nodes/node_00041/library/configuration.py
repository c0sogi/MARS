import os
import torch


class Config:
    """
    Configuration class for the Heterogeneous Mutual-Distillation Pipeline.
    Centralizes all hyperparameters, paths, and execution settings.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    seed = 42
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Debugging: Set to True to run on a small subset of data
    debug = False
    debug_sample_size = 100

    # =========================================================================
    # Directories and Paths
    # =========================================================================
    # Input Data (Metadata)
    input_dir = "./metadata"
    train_path = os.path.join(input_dir, "train.csv")
    val_path = os.path.join(input_dir, "validation.csv")
    test_path = os.path.join(input_dir, "test.csv")

    # Working Directory (Artifacts)
    working_dir = "./working/idea_12"
    output_dir = os.path.join(working_dir, "outputs")
    cache_dir = os.path.join(working_dir, "cache")

    # Submission
    submission_dir = "./submission"
    submission_path = os.path.join(submission_dir, "submission.csv")

    # Ensure necessary directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(submission_dir, exist_ok=True)

    # =========================================================================
    # Data Processing & Feature Engineering
    # =========================================================================
    max_len = 256  # Maximum sequence length for tokenizers

    # SVD Structural Features
    svd_components = 256
    tfidf_word_ngram_range = (1, 2)
    tfidf_char_ngram_range = (3, 5)

    # =========================================================================
    # Model Architectures
    # =========================================================================
    # Stream A: DeBERTa
    model_a_name = "microsoft/deberta-v3-large"

    # Stream B: RoBERTa
    model_b_name = "roberta-large"

    # Head Configuration
    hidden_size = 1024  # Hidden size for large transformer models
    svd_embedding_size = 256  # Matches svd_components

    # Variable-Rate Multi-Sample Dropout (VR-MSD)
    dropout_rates = [0.1, 0.2, 0.3, 0.4, 0.5]

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    num_folds = 5
    epochs = 4

    # Batch Sizes (Adjusted for A100 40GB)
    train_batch_size = 8
    valid_batch_size = 16
    gradient_accumulation_steps = 1
    max_grad_norm = 1.0

    # Differential Learning Rates
    lr_backbone = 1e-5
    lr_head = 1e-3
    weight_decay = 0.01

    # Scheduler
    scheduler_type = "cosine"
    warmup_ratio = 0.1

    # Adversarial Weight Perturbation (AWP)
    use_awp = True
    awp_lr = 1e-4
    awp_eps = 1e-2
    awp_start_epoch = 1  # Enable AWP after the first epoch

    # =========================================================================
    # Distillation Settings
    # =========================================================================
    distillation_temp = 2.0
    distillation_alpha = 0.5  # Weight for Soft Targets (vs Hard Targets)

    # =========================================================================
    # Helper Methods
    # =========================================================================
    @classmethod
    def get_checkpoint_path(cls, model_name, fold, stage="teacher"):
        """
        Generates a standardized path for saving/loading model checkpoints.

        Args:
            model_name (str): Name of the backbone (e.g., 'roberta-large').
            fold (int): Fold number.
            stage (str): 'teacher' or 'student'.

        Returns:
            str: Full path to the checkpoint file.
        """
        clean_name = model_name.replace("/", "_").replace("-", "_")
        filename = f"{stage}_{clean_name}_fold_{fold}.bin"
        return os.path.join(cls.output_dir, filename)

    @classmethod
    def get_svd_cache_path(cls, dataset_name):
        """
        Generates path for cached SVD features.

        Args:
            dataset_name (str): 'train', 'val', or 'test'.
        """
        return os.path.join(cls.cache_dir, f"{dataset_name}_svd_features.npy")
