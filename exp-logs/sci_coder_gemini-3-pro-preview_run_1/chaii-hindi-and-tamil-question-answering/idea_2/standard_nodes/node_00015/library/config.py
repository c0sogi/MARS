import os
import torch


class Config:
    """
    Configuration class for the Extractive QA Retriever-Reader model.
    Centralizes hyperparameters, file paths, and model settings.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    seed = 42
    # Debugging: Set to True to train/predict on a small subset of data
    debug = False
    debug_subset_size = 50

    # =========================================================================
    # File Paths
    # =========================================================================
    # Input Data (Generated in ./metadata as per instructions)
    train_path = "./metadata/train.csv"
    val_path = "./metadata/val.csv"
    test_path = "./metadata/test.csv"
    sample_submission_path = "./input/sample_submission.csv"

    # Output Directories
    working_dir = "./working/idea_2/"
    model_output_dir = os.path.join(working_dir, "best_model")
    submission_path = "./submission/submission.csv"

    # Cache Files (for deterministic data processing)
    train_cache_path = os.path.join(working_dir, "train_processed.parquet")
    val_cache_path = os.path.join(working_dir, "val_processed.parquet")
    test_cache_path = os.path.join(working_dir, "test_processed.parquet")

    # Ensure output directories exist
    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)

    # =========================================================================
    # Model Architecture
    # =========================================================================
    model_checkpoint = "xlm-roberta-base"

    # =========================================================================
    # Input Processing (Tokenizer & Sliding Window)
    # =========================================================================
    max_length = 384  # Maximum sequence length (Question + Context)
    doc_stride = 128  # Overlap between sliding windows

    # =========================================================================
    # Data Strategy
    # =========================================================================
    # Ratio of Negative (No-Answer) windows to Positive windows to keep during training.
    # 2.0 means we keep approximately 2 negative samples for every 1 positive sample.
    negative_sampling_ratio = 2.0

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    epochs = 3
    train_batch_size = 16
    eval_batch_size = 32
    learning_rate = 2e-5
    weight_decay = 0.01
    warmup_ratio = 0.1
    max_grad_norm = 1.0

    # =========================================================================
    # Inference & Post-Processing
    # =========================================================================
    n_best_size = 20  # Number of logits to consider for start/end positions
    max_answer_length = 30  # Maximum allowed length for a predicted answer

    # =========================================================================
    # Hardware
    # =========================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 4
