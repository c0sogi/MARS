import os
import torch


class Config:
    """
    Configuration class for Whale Identification Task.
    Implements settings for EfficientNet-B2 + AdaFace pipeline.
    """

    # -------------------------------------------------------------------------
    # General & Hardware
    # -------------------------------------------------------------------------
    seed = 42
    debug = False  # Set to True to run on a small subset for testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 8  # Optimized for 12 vCPUs

    # -------------------------------------------------------------------------
    # Directories & Paths
    # -------------------------------------------------------------------------
    input_dir = "./input"
    metadata_dir = "./metadata"
    working_dir = "./working/idea_4"

    # Ensure working directory exists
    os.makedirs(working_dir, exist_ok=True)

    # Input Metadata
    train_csv_path = os.path.join(metadata_dir, "train.csv")
    val_csv_path = os.path.join(metadata_dir, "val.csv")
    test_csv_path = os.path.join(metadata_dir, "test.csv")

    # Output Artifacts
    model_save_path = os.path.join(working_dir, "efficientnet_b2_adaface.pth")
    submission_path = os.path.join(working_dir, "submission.csv")

    # Cache Paths (using .npy as requested for deterministic caching)
    # These are used to store embeddings for retrieval/validation
    train_embeddings_path = os.path.join(working_dir, "train_embeddings.npy")
    train_labels_path = os.path.join(working_dir, "train_labels.npy")
    val_embeddings_path = os.path.join(working_dir, "val_embeddings.npy")
    val_labels_path = os.path.join(working_dir, "val_labels.npy")
    test_embeddings_path = os.path.join(working_dir, "test_embeddings.npy")
    test_names_path = os.path.join(working_dir, "test_names.npy")

    # -------------------------------------------------------------------------
    # Model Architecture
    # -------------------------------------------------------------------------
    model_name = "efficientnet_b2"
    input_size = 448  # Resolution: 448x448
    embedding_dim = 512
    drop_rate = 0.3

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    # Aggressive optimization strategy
    batch_size = 16
    epochs = 20
    learning_rate = 1e-3
    weight_decay = 1e-4  # Standard for AdamW

    # Scheduling & Stopping
    scheduler_patience = 2
    early_stopping_patience = 5

    # -------------------------------------------------------------------------
    # AdaFace Head Hyperparameters
    # -------------------------------------------------------------------------
    # Scale factor (s)
    adaface_scale = 64.0
    # Margin (m)
    adaface_margin = 0.5
    # Head concentration parameter (h) - specific to AdaFace
    adaface_h = 0.333

    # -------------------------------------------------------------------------
    # Data Processing
    # -------------------------------------------------------------------------
    exclude_new_whale = True  # Exclude 'new_whale' from training loop

    # -------------------------------------------------------------------------
    # Inference & Post-Processing
    # -------------------------------------------------------------------------
    # k-NN Retrieval
    knn_k = 100

    # Query Expansion (QE)
    use_qe = True
    qe_k = 5  # Number of neighbors for average query expansion

    # Open-Set Recognition
    # Similarity threshold: if top match similarity < threshold, predict 'new_whale'
    # Assuming Cosine Similarity (-1 to 1)
    unknown_threshold = 0.45

    @classmethod
    def print_config(cls):
        """Prints the current configuration."""
        print("\n" + "=" * 40)
        print(f"{'CONFIG':^40}")
        print("=" * 40)
        for k, v in cls.__dict__.items():
            if not k.startswith("__") and not callable(v):
                print(f"{k:<25} : {v}")
        print("=" * 40 + "\n")
