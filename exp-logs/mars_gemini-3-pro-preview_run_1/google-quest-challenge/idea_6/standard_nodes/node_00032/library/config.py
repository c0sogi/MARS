import os
import torch


class Config:
    # ==========================================
    # General Settings
    # ==========================================
    seed = 42
    debug = False  # Set to True to run on a small subset for debugging
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_freq = 100  # Logging frequency

    # ==========================================
    # Paths
    # ==========================================
    # Base directories
    input_dir = "./input"
    metadata_dir = "./metadata"
    working_dir = "./working/idea_6"
    submission_dir = "./submission"

    # Ensure output directories exist
    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(submission_dir, exist_ok=True)

    # Data file paths
    train_path = os.path.join(metadata_dir, "train.csv")
    val_path = os.path.join(metadata_dir, "val.csv")
    # Cite debug_lesson_2: Use raw input file for test to handle runtime swapping
    test_path = os.path.join(input_dir, "test.csv")
    sample_submission_path = os.path.join(input_dir, "sample_submission.csv")

    # Output paths
    submission_path = os.path.join(submission_dir, "submission.csv")
    model_save_path = os.path.join(working_dir, "best_model.pth")

    # Caching paths (Parquet format)
    train_cache_path = os.path.join(working_dir, "train_cached.parquet")
    val_cache_path = os.path.join(working_dir, "val_cached.parquet")
    test_cache_path = os.path.join(working_dir, "test_cached.parquet")

    # ==========================================
    # Model Architecture
    # ==========================================
    model_name = "microsoft/deberta-v3-small"
    hidden_size = 768  # Hidden size for deberta-v3-small
    num_classes = 30  # Number of target labels
    max_len = 512  # Maximum sequence length

    # Architecture Specifics (Idea 6)
    use_partitioned_pooling = (
        True  # Masked Global Avg + Max Pooling for Q and A separately
    )
    use_interaction_fusion = True  # Include interaction terms (u*v, |u-v|)
    use_nonlinear_head = (
        True  # Use MLP (Linear-GELU-Dropout-Linear) instead of simple Linear
    )

    # Dropout
    dropout_rate = 0.1
    # Multi-sample dropout rates for the head to improve generalization
    multi_sample_dropout_rates = [0.1, 0.2, 0.3, 0.4, 0.5]

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    epochs = 5
    train_batch_size = 16
    valid_batch_size = 32  # Larger batch size for inference/validation

    # Optimization
    backbone_lr = 2e-5  # Lower LR for pre-trained weights
    head_lr = 1e-3  # Higher LR for the new head
    weight_decay = 0.01
    eps = 1e-6
    betas = (0.9, 0.999)
    max_grad_norm = 1.0  # Gradient clipping

    # Scheduler
    scheduler_type = "linear"  # Linear decay with warmup
    warmup_ratio = 0.1  # Warmup for 10% of total steps

    # ==========================================
    # Data Processing
    # ==========================================
    load_cached_data = True  # Toggle to load from cache if available

    # Column mappings
    question_title_col = "question_title"
    question_body_col = "question_body"
    answer_col = "answer"
    qa_id_col = "qa_id"
