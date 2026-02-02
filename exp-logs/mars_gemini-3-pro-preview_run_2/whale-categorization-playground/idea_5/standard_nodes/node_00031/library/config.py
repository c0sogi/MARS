import os
import torch


class Config:
    # ---------------------------------------------------------
    # General Configuration
    # ---------------------------------------------------------
    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 10  # Optimized for 12 vCPUs

    # ---------------------------------------------------------
    # Directory Paths
    # ---------------------------------------------------------
    input_dir = "./input"
    metadata_dir = "./metadata"
    working_dir = "./working/idea_5"
    submission_dir = "./submission"

    # Create necessary directories
    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(submission_dir, exist_ok=True)

    # ---------------------------------------------------------
    # Metadata Paths
    # ---------------------------------------------------------
    train_csv_path = os.path.join(metadata_dir, "train.csv")
    val_csv_path = os.path.join(metadata_dir, "val.csv")
    test_csv_path = os.path.join(metadata_dir, "test.csv")
    sample_submission_path = os.path.join(input_dir, "sample_submission.csv")
    submission_path = os.path.join(submission_dir, "submission.csv")

    # ---------------------------------------------------------
    # Caching Paths (Parameterized for Invalidation)
    # ---------------------------------------------------------
    # Suffix ensures we don't load data from different resolutions/backbones
    cache_suffix = "b4_384"

    # Numpy cache files
    train_images_cache = os.path.join(working_dir, f"train_images_{cache_suffix}.npy")
    train_labels_cache = os.path.join(working_dir, f"train_labels_{cache_suffix}.npy")

    val_images_cache = os.path.join(working_dir, f"val_images_{cache_suffix}.npy")
    val_labels_cache = os.path.join(working_dir, f"val_labels_{cache_suffix}.npy")

    test_images_cache = os.path.join(working_dir, f"test_images_{cache_suffix}.npy")
    test_ids_cache = os.path.join(working_dir, f"test_ids_{cache_suffix}.npy")

    # ---------------------------------------------------------
    # Model Architecture
    # ---------------------------------------------------------
    backbone = "tf_efficientnet_b4"
    image_size = 384
    embedding_size = 512
    dropout_rate = 0.3

    # Gradient Checkpointing is CRITICAL for B4 @ 384px with Batch Size 32
    use_gradient_checkpointing = True
    pretrained = True

    # ---------------------------------------------------------
    # Training Hyperparameters
    # ---------------------------------------------------------
    batch_size = 32
    epochs = 25
    learning_rate = 3e-4
    weight_decay = 1e-4

    # Learning Rate Scheduler
    scheduler_patience = 2
    scheduler_factor = 0.5
    min_lr = 1e-6

    # Early Stopping
    early_stopping_patience = 5

    # ArcFace Hyperparameters
    arc_s = 30.0
    arc_m = 0.50

    # ---------------------------------------------------------
    # Inference & Evaluation
    # ---------------------------------------------------------
    # Threshold for deciding if a whale is "new_whale"
    # If distance to nearest known whale > threshold (or similarity < threshold), predict new_whale
    # Note: ArcFace uses Cosine Similarity.
    # We will use a similarity threshold.
    new_whale_threshold = 0.45

    model_save_path = os.path.join(working_dir, "best_model.pth")
