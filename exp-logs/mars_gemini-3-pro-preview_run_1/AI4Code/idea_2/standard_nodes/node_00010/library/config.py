import os


class Config:
    # Random Seed for reproducibility
    seed = 42

    # Model Hyperparameters
    model_name = "all-MiniLM-L6-v2"  # Sentence Transformer backbone
    input_dim = 384  # Dimension of MiniLM embeddings
    projection_dim = 512  # Dimension of the shared latent space
    hidden_dim = 512  # Hidden dimension for internal layers
    dropout = 0.1  # Dropout rate

    # Training Hyperparameters
    lr = 1e-3  # Aggressive constant learning rate
    batch_size = 64  # Batch size
    num_epochs = 5  # Maximum number of epochs
    patience = 3  # Early stopping patience

    # Data Processing
    max_length = 128  # Max token length for sentence transformer
    num_workers = 4  # Number of dataloader workers

    # Directories
    input_dir = "./input"
    metadata_dir = "./metadata"
    working_dir = "./working/idea_2"
    submission_dir = "./submission"

    # File Paths - Metadata
    train_metadata_path = os.path.join(metadata_dir, "train_metadata.csv")
    val_metadata_path = os.path.join(metadata_dir, "val_metadata.csv")
    test_metadata_path = os.path.join(metadata_dir, "test_metadata.csv")

    # File Paths - Cached Features (Parquet)
    train_features_path = os.path.join(working_dir, "train_features.parquet")
    val_features_path = os.path.join(working_dir, "val_features.parquet")
    test_features_path = os.path.join(working_dir, "test_features.parquet")

    # File Paths - Model & Output
    model_save_path = os.path.join(working_dir, "best_model.pth")
    submission_path = os.path.join(submission_dir, "submission.csv")

    @classmethod
    def setup(cls):
        """
        Ensures necessary directories exist.
        """
        os.makedirs(cls.working_dir, exist_ok=True)
        os.makedirs(cls.submission_dir, exist_ok=True)


# Automatically create directories when config is imported
Config.setup()
