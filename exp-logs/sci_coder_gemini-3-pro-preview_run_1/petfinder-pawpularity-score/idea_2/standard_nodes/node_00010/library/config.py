import os
import torch


class Config:
    # =========================================================================
    # General Settings
    # =========================================================================
    seed = 42
    debug = False  # Set to True to run with a small subset of data for debugging
    debug_sample_size = 100  # Number of samples to use when debug=True
    num_workers = 4  # Number of dataloader workers

    # =========================================================================
    # File Paths & Directories
    # =========================================================================
    # Root directory for read-only input data
    input_root = "./input"

    # Directory containing the generated metadata CSVs
    metadata_dir = "./metadata"

    # Working directory for caching intermediate files (features, models, etc.)
    working_dir = "./working/idea_2"

    # Directory for final submission output
    submission_dir = "./submission"

    # Specific file paths
    train_metadata_path = os.path.join(metadata_dir, "train.csv")
    val_metadata_path = os.path.join(metadata_dir, "val.csv")
    test_metadata_path = os.path.join(metadata_dir, "test.csv")

    # Output paths
    model_save_path = os.path.join(working_dir, "best_model.pth")
    submission_path = os.path.join(submission_dir, "submission.csv")

    # Ensure writable directories exist
    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(submission_dir, exist_ok=True)

    # =========================================================================
    # Data Configuration
    # =========================================================================
    target_col = "Pawpularity"

    # Feature columns based on Data Analysis (matches CSV headers)
    # Note: 'Subject Focus' is used instead of 'Focus' as seen in analysis output
    feature_cols = [
        "Subject Focus",
        "Eyes",
        "Face",
        "Near",
        "Action",
        "Accessory",
        "Group",
        "Collage",
        "Human",
        "Occlusion",
        "Info",
        "Blur",
    ]

    # Image parameters
    img_size = 224

    # =========================================================================
    # Model Architecture
    # =========================================================================
    # Swin Transformer Small backbone (Cite solution_lesson_node_00006)
    model_name = "swin_small_patch4_window7_224"

    # MLP Head configuration
    fc_dim = 128
    dropout = 0.1

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    epochs = 8
    batch_size = 32

    # Optimization
    learning_rate = 2e-5  # Lower learning rate for fine-tuning
    weight_decay = 1e-2
    max_grad_norm = 10.0

    # Scheduler (Cosine Annealing)
    T_max = epochs
    min_lr = 1e-6

    # Early Stopping
    early_stopping_patience = 3
    early_stopping_mode = "min"  # Monitor RMSE (minimize)

    # =========================================================================
    # Hardware
    # =========================================================================
    device = "cuda" if torch.cuda.is_available() else "cpu"
