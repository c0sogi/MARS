import os
import torch


class Config:
    # General
    seed = 42
    debug = False  # Set to True to run with a small subset of data
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data Paths
    input_dir = "./input"
    metadata_dir = "./metadata"
    working_dir = "./working/idea_3"
    submission_dir = "./submission"

    # Specific File Paths
    train_metadata_path = os.path.join(metadata_dir, "train.csv")
    val_metadata_path = os.path.join(metadata_dir, "val.csv")
    test_metadata_path = os.path.join(metadata_dir, "test.csv")

    train_raw_path = os.path.join(input_dir, "train.csv")
    test_raw_path = os.path.join(input_dir, "test.csv")
    sample_submission_path = os.path.join(input_dir, "sample_submission.csv")

    # Output Paths
    model_save_path = os.path.join(working_dir, "model.pth")
    submission_path = os.path.join(submission_dir, "submission.csv")

    # Model Architecture
    model_name = "microsoft/deberta-v3-base"
    max_len = 300
    n_dropout_samples = 5  # Multi-sample dropout count
    dropout_rate = 0.1
    num_classes = 6

    # Training Hyperparameters
    epochs = 4
    train_batch_size = 16
    valid_batch_size = 32
    learning_rate = 2e-5
    weight_decay = 0.01
    max_grad_norm = 1.0

    # Scheduler
    scheduler_type = "OneCycleLR"
    pct_start = 0.1  # Percentage of training to increase LR

    def __init__(self):
        # Ensure working and submission directories exist
        os.makedirs(self.working_dir, exist_ok=True)
        os.makedirs(self.submission_dir, exist_ok=True)

    @property
    def labels(self):
        return [
            "toxic",
            "severe_toxic",
            "obscene",
            "threat",
            "insult",
            "identity_hate",
        ]
