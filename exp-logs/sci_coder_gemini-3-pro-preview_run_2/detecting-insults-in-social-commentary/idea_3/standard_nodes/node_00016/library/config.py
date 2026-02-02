import os
import torch


class Config:
    # General Setup
    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 4

    # Data Paths
    input_dir = "./input"
    metadata_dir = "./metadata"
    train_path = os.path.join(metadata_dir, "train.csv")
    val_path = os.path.join(metadata_dir, "val.csv")
    test_path = os.path.join(metadata_dir, "test.csv")

    # Output Paths
    working_dir = "./working/idea_4"
    submission_dir = "./submission"
    submission_path = os.path.join(submission_dir, "submission.csv")

    # Model Architecture
    model_name = "microsoft/deberta-v3-large"
    max_len = 128
    dropout_samples = 5  # Number of dropout branches for Multi-Sample Dropout

    # Training Hyperparameters
    n_folds = 5
    epochs = 5
    train_batch_size = 8
    valid_batch_size = 16
    learning_rate = 1e-5
    weight_decay = 0.01
    max_grad_norm = 1000

    # Scheduler & Optimization
    scheduler = "cosine"  # 'linear' or 'cosine'
    warmup_ratio = 0.1
    llrd_decay = 0.9  # Layer-wise learning rate decay factor

    # Early Stopping
    patience = 3  # Stop if validation AUC doesn't improve for 3 epochs

    # Logging
    print_freq = 50  # Print training status every N steps

    @classmethod
    def setup(cls):
        """Creates necessary directories."""
        os.makedirs(cls.working_dir, exist_ok=True)
        os.makedirs(cls.submission_dir, exist_ok=True)


# Initialize directories immediately when config is imported
Config.setup()
