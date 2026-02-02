import os
import torch
import random
import numpy as np


class Config:
    # ==== General Settings ====
    seed = 42
    debug = False  # Set to True to run on a small subset for debugging
    debug_subset_size = 100

    # ==== Data Paths ====
    # Input data (Read-Only)
    input_dir = "./input"
    # Metadata (Generated splits)
    metadata_dir = "./metadata"
    train_path = os.path.join(metadata_dir, "train.csv")
    val_path = os.path.join(metadata_dir, "val.csv")
    test_path = os.path.join(metadata_dir, "test.csv")
    sample_submission_path = os.path.join(input_dir, "sample_submission.csv")

    # Working Directory (Write access)
    working_dir = "./working/idea_9"
    output_dir = os.path.join(working_dir, "output")
    cache_dir = os.path.join(working_dir, "cache")
    model_save_path = os.path.join(working_dir, "best_model.pth")
    submission_path = "./submission/submission.csv"

    # Ensure directories exist
    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)

    # ==== Model Architecture ====
    model_name = "microsoft/deberta-v3-large"
    max_length = 512
    num_classes = 3  # Winner A, Winner B, Tie

    # Pooling & Head Configuration
    # We use the last 4 layers for pooling
    num_pooling_layers = 4
    # Dropout rate for the classification head
    dropout = 0.1

    # ==== Training Hyperparameters ====
    # Physical batch size (what fits in GPU memory)
    # A100 40GB can typically handle 4-8 for DeBERTa-Large with gradient checkpointing.
    train_batch_size = 4
    valid_batch_size = 8

    # Target effective batch size (achieved via gradient accumulation)
    target_batch_size = 64

    # Calculate accumulation steps
    gradient_accumulation_steps = max(1, target_batch_size // train_batch_size)

    epochs = 3
    learning_rate = 5e-6
    weight_decay = 0.01
    eps = 1e-6

    # Scheduler
    scheduler_type = "cosine"
    warmup_ratio = 0.1

    # Optimization flags
    use_fp16 = True  # Mixed precision
    grad_checkpointing = True  # Enable to save memory for Large model
    max_grad_norm = 1.0

    # ==== System ====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 4  # Based on 12 vCPUs available


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Apply seed immediately upon import
seed_everything(Config.seed)
