import os
import torch


class CFG:
    """
    Configuration class for the Patent Phrase Similarity task.
    Centralizes all hyperparameters, paths, and model settings.
    """

    # ====================================================
    # General Settings
    # ====================================================
    seed = 42
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    debug = False  # Set to True to run on a small subset for debugging
    print_freq = 100  # Logging frequency in steps

    # ====================================================
    # Paths
    # ====================================================
    # Input directories (Read-Only)
    input_dir = "./input"
    metadata_dir = "./metadata"

    # Output directories (Write-Accessible)
    # Using 'idea_8' as the working directory for this specific iteration
    working_dir = "./working/idea_8"
    submission_dir = "./submission"

    # File paths
    train_file = os.path.join(metadata_dir, "train.csv")
    val_file = os.path.join(metadata_dir, "val.csv")
    test_file = os.path.join(metadata_dir, "test.csv")
    sample_submission = os.path.join(input_dir, "sample_submission.csv")

    # Raw CPC description file
    cpc_description_file = os.path.join(input_dir, "description.md")

    # Cache paths for processed data
    cpc_cache_path = os.path.join(working_dir, "cpc_texts.parquet")

    # Ensure output directories exist
    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(submission_dir, exist_ok=True)

    # ====================================================
    # Model Architecture
    # ====================================================
    model_name = "microsoft/deberta-v3-large"
    gradient_checkpointing = True

    # Weighted Layer Pooling (WLP)
    # We will pool the [CLS] embeddings from the last N hidden layers
    num_pooling_layers = 4

    # Multi-Sample Dropout (MSD)
    # Number of dropout masks to apply in the classification head
    num_msd = 5
    fc_dropout = 0.2

    # ====================================================
    # Tokenizer
    # ====================================================
    # Max length to accommodate: Context Description + Anchor + Target
    max_len = 133

    # ====================================================
    # Training Hyperparameters
    # ====================================================
    n_folds = 5
    num_epochs = 4

    # Batch Size & Accumulation
    # A100 40GB allows for decent batch sizes with DeBERTa-Large.
    # Effective Batch Size = batch_size * accum_iter
    batch_size = 16
    accum_iter = 1

    # Optimizer (AdamW)
    learning_rate = 2e-5
    weight_decay = 0.01
    eps = 1e-6
    betas = (0.9, 0.999)
    max_grad_norm = 1000

    # Scheduler (Cosine with Warmup)
    scheduler_type = "cosine"
    warmup_ratio = 0.1
    num_cycles = 0.5

    # Layer-wise Learning Rate Decay (LLRD)
    # Lower layers get smaller learning rates to preserve pre-trained knowledge
    llrd_decay = 0.9

    # ====================================================
    # EMA (Exponential Moving Average)
    # ====================================================
    # Stabilizes training and improves generalization
    use_ema = True
    ema_decay = 0.999
    ema_update_every = 1

    # ====================================================
    # Inference
    # ====================================================
    inference_batch_size = 32
