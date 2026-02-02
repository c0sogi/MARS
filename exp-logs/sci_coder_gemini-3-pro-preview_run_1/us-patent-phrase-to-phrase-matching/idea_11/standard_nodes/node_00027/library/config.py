import os
import torch


class Config:
    # =========================================================================
    # General Settings
    # =========================================================================
    seed = 42
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    debug = False  # Set to True to run on a small subset for debugging
    debug_sample_size = 100  # Number of samples to use in debug mode

    # =========================================================================
    # File Paths
    # =========================================================================
    # Input directories
    input_dir = "./input"
    metadata_dir = "./metadata"

    # Specific file paths based on metadata generation
    train_path = os.path.join(metadata_dir, "train.csv")
    val_path = os.path.join(metadata_dir, "val.csv")
    test_path = os.path.join(metadata_dir, "test.csv")

    # Raw data for context expansion
    cpc_codes_path = os.path.join(input_dir, "description.md")
    sample_submission_path = os.path.join(input_dir, "sample_submission.csv")

    # Output and Caching
    # We use idea_11 as the specific working directory for this strategy
    working_dir = "./working/idea_11"
    output_dir = working_dir
    cache_dir = working_dir

    # Ensure working directory exists
    os.makedirs(working_dir, exist_ok=True)

    # =========================================================================
    # Model Architecture
    # =========================================================================
    model_name = "microsoft/deberta-v3-large"
    # Max length covers: [CLS] Context (hierarchical) [SEP] Anchor [SEP] Target [SEP]
    # 133 is generally sufficient for this dataset with full context expansion
    max_length = 133

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    epochs = 4
    # Batch size of 16 fits comfortably on A100 40GB with FP16 for DeBERTa-Large
    train_batch_size = 16
    valid_batch_size = 32

    # Optimization
    learning_rate = 2e-5
    weight_decay = 0.01
    max_grad_norm = 1000.0

    # Scheduler
    scheduler_type = "cosine"
    warmup_ratio = 0.1

    # Precision
    fp16 = True

    # =========================================================================
    # Advanced Strategy Settings
    # =========================================================================
    # Layer-wise Learning Rate Decay (LLRD)
    # Decay rate for layers as we go from top (head) to bottom (embeddings)
    llrd_decay = 0.9

    # Loss Function
    # Using MSE Loss as it aligns geometrically with Pearson correlation
    loss_type = "mse"

    # =========================================================================
    # Cross-Validation
    # =========================================================================
    num_folds = 5
    target_col = "score"
    group_col = "anchor"  # For GroupKFold to prevent leakage
