import os
import torch


class Config:
    # --- General ---
    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 4
    debug = False  # Set to True to run on a small subset for debugging
    debug_subset_size = 100

    # --- Paths ---
    # Input paths (using the generated metadata)
    train_path = "./metadata/train.csv"
    val_path = "./metadata/val.csv"
    test_path = "./metadata/test.csv"
    sample_submission_path = "./input/sample_submission.csv"

    # Output paths
    output_dir = "./working/idea_3"
    model_dir = os.path.join(output_dir, "models")
    submission_path = "./submission/submission.csv"

    # Ensure output directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # --- Model (Semantic Branch) ---
    model_name = "microsoft/deberta-v3-large"
    max_length = 1024
    dropout = (
        0.0  # No dropout for the regression head typically yields better stability
    )
    target_cols = ["score"]

    # --- Model (Lexical Branch) ---
    # TF-IDF + Ridge settings
    tfidf_ngram_range = (1, 3)
    tfidf_min_df = 3
    tfidf_max_features = 50000  # Cap features to prevent memory explosion

    # --- Training ---
    folds = 5
    epochs = 4
    train_batch_size = 4  # DeBERTa-large with 1024 tokens is memory intensive
    valid_batch_size = 8
    gradient_accumulation_steps = 1
    max_grad_norm = 10.0

    # Optimization
    learning_rate = 1e-5  # Uniform learning rate
    weight_decay = 0.01
    scheduler = "cosine"  # 'linear' or 'cosine'
    warmup_ratio = 0.0  # Small warmup
    early_stopping_patience = 3

    # Loss
    use_smooth_l1 = True  # Use SmoothL1Loss instead of MSE

    # Inference
    use_fp16 = True  # Use mixed precision
