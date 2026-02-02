import os
import torch


class Config:
    """
    Configuration class for Fine-Tuned DistilRoBERTa with Distribution-Aware Thresholding.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 4  # Number of subprocesses for data loading

    # Debugging mode: if True, restricts data to a small subset for testing
    debug = False
    debug_sample_size = 50000

    # =========================================================================
    # File Paths
    # =========================================================================
    # Input Data (Metadata)
    metadata_dir = "./metadata"
    train_path = os.path.join(metadata_dir, "train.csv")
    val_path = os.path.join(metadata_dir, "validation.csv")
    test_path = os.path.join(metadata_dir, "test.csv")

    # Output / Working Directory
    working_dir = "./working/idea_4"
    os.makedirs(working_dir, exist_ok=True)

    # Artifacts
    # We use JSON for the tag vocabulary to avoid pickle
    tags_path = os.path.join(working_dir, "tags.json")
    # Path to save the best model state dict
    model_save_path = os.path.join(working_dir, "distilroberta_finetuned.pth")
    # Path to cache processed datasets (if needed by data loader)
    cache_dir = os.path.join(working_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Submission
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)
    submission_path = os.path.join(submission_dir, "submission.csv")

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================
    model_name = "distilroberta-base"
    max_length = 128  # Truncate to 128 tokens for efficiency on 4M+ samples
    num_labels = 5000  # Number of most frequent tags to predict
    dropout_rate = 0.1
    hidden_size = 768  # DistilRoBERTa hidden size

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    epochs = 1
    train_batch_size = 128
    valid_batch_size = 256
    learning_rate = 3e-5
    weight_decay = 0.01
    warmup_ratio = 0.1
    max_grad_norm = 1.0

    # Mixed Precision
    use_fp16 = True

    # =========================================================================
    # Inference / Thresholding
    # =========================================================================
    # Thresholds will be determined dynamically based on validation distribution.
    # We define the percentile range to search for the optimal threshold.
    threshold_search_start_percentile = 50
    threshold_search_end_percentile = 99
    threshold_search_steps = 100
