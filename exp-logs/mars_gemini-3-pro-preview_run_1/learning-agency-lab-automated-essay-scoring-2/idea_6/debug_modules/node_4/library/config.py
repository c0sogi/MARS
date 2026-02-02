import os
import torch


class Config:
    """
    Configuration class for the Essay Scoring task.
    Centralizes all hyperparameters, file paths, and model settings for the
    Domain-Adaptive Adversarial Pipeline.
    """

    # ==========================
    # General Settings
    # ==========================
    seed = 42
    debug = False  # Set to True to run on a small subset for debugging
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_freq = 50

    # ==========================
    # Paths
    # ==========================
    # Input directories (Read-Only)
    input_dir = "./input"
    metadata_dir = "./metadata"

    # Output directory (Read/Write)
    # Using 'idea_6' to isolate this experiment's artifacts
    working_dir = "./working/idea_6"

    # Data Paths (from Metadata)
    train_path = os.path.join(metadata_dir, "train.csv")
    val_path = os.path.join(metadata_dir, "val.csv")
    test_path = os.path.join(metadata_dir, "test.csv")
    sample_submission_path = os.path.join(input_dir, "sample_submission.csv")

    # Cache Paths for processed data (Parquet format)
    train_cache_path = os.path.join(working_dir, "train_processed.parquet")
    val_cache_path = os.path.join(working_dir, "val_processed.parquet")
    test_cache_path = os.path.join(working_dir, "test_processed.parquet")
    mlm_data_cache_path = os.path.join(working_dir, "mlm_data.parquet")

    # Model Checkpoint Paths
    mlm_model_dir = os.path.join(working_dir, "mlm_checkpoints")
    model_save_path = os.path.join(working_dir, "model.pth")
    submission_path = os.path.join(working_dir, "submission.csv")

    # ==========================
    # Model Architecture
    # ==========================
    model_name = "microsoft/deberta-v3-large"
    max_length = 1024
    num_labels = 1  # Regression output
    dropout = 0.0  # Dropout probability

    # ==========================
    # Training: Stage 1 - Masked Language Modeling (MLM)
    # ==========================
    mlm_epochs = 3
    mlm_batch_size = 4
    mlm_learning_rate = 5e-5
    mlm_mask_probability = 0.15
    mlm_weight_decay = 0.01

    # ==========================
    # Training: Stage 2 - Supervised Fine-Tuning (SFT)
    # ==========================
    n_folds = 5
    train_batch_size = 4
    valid_batch_size = 8
    gradient_accumulation_steps = 4
    epochs = 4

    # Optimization
    learning_rate = 1e-5
    weight_decay = 0.01
    max_grad_norm = 1000.0
    warmup_ratio = 0.1
    scheduler = "cosine"  # Options: 'cosine', 'linear', 'constant'

    # Layer-wise Learning Rate Decay (LLRD)
    llrd_decay = 0.9

    # ==========================
    # Adversarial Weight Perturbation (AWP)
    # ==========================
    use_awp = True
    awp_lr = 1e-4
    awp_eps = 1e-2
    awp_start_epoch = 1  # Epoch to start AWP (e.g., after the 1st epoch)

    # ==========================
    # Data Information
    # ==========================
    target_col = "score"
    text_col = "full_text"
    id_col = "essay_id"

    # ==========================
    # Setup Logic
    # ==========================
    @classmethod
    def setup(cls):
        """
        Ensures necessary working directories exist.
        """
        os.makedirs(cls.working_dir, exist_ok=True)
        os.makedirs(cls.mlm_model_dir, exist_ok=True)


# Initialize directories on import
Config.setup()
