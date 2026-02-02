import os
import torch
import random
import numpy as np


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Config:
    # -------------------------------------------------------------------------
    # General Configuration
    # -------------------------------------------------------------------------
    seed = 42
    debug = False  # Set to True to use a small subset of data for debugging
    debug_sample_size = 100

    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------------------
    # Data Paths
    # -------------------------------------------------------------------------
    input_dir = "./input"
    train_dir = os.path.join(input_dir, "train")
    test_dir = os.path.join(input_dir, "test")

    meta_dir = "./metadata"
    meta_train_path = os.path.join(meta_dir, "train.csv")
    meta_val_path = os.path.join(meta_dir, "val.csv")
    meta_test_path = os.path.join(meta_dir, "test.csv")

    # -------------------------------------------------------------------------
    # Working Directory & Outputs
    # -------------------------------------------------------------------------
    working_dir = "./working/idea_3"
    os.makedirs(working_dir, exist_ok=True)

    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)
    submission_path = os.path.join(submission_dir, "submission.csv")

    model_path = os.path.join(working_dir, "best_model.pth")

    # -------------------------------------------------------------------------
    # Caching (Parameterized by Resolution)
    # -------------------------------------------------------------------------
    # We use .npy format for efficient loading of preprocessed tensors
    img_size = 384

    # Train Cache
    cache_train_images = os.path.join(working_dir, f"train_images_{img_size}.npy")
    cache_train_labels = os.path.join(working_dir, "train_labels.npy")
    cache_train_ids = os.path.join(working_dir, "train_ids.npy")

    # Val Cache
    cache_val_images = os.path.join(working_dir, f"val_images_{img_size}.npy")
    cache_val_labels = os.path.join(working_dir, "val_labels.npy")
    cache_val_ids = os.path.join(working_dir, "val_ids.npy")

    # Test Cache
    cache_test_images = os.path.join(working_dir, f"test_images_{img_size}.npy")
    cache_test_ids = os.path.join(working_dir, "test_ids.npy")

    # -------------------------------------------------------------------------
    # Model Architecture
    # -------------------------------------------------------------------------
    backbone = "convnext_tiny"
    embedding_size = 512

    # Number of known identities (Total 4029 - 1 'new_whale' = 4028)
    # The 'new_whale' class is excluded from the classification head during training.
    n_classes = 4028

    # ArcFace Hyperparameters
    arcface_s = 30.0
    arcface_m = 0.50

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    epochs = 15
    train_batch_size = 24  # Adjusted for 384x384 on GPU
    valid_batch_size = 32

    learning_rate = 3e-4
    min_lr = 1e-6
    weight_decay = 1e-6
    scheduler_T_max = 15

    # -------------------------------------------------------------------------
    # Inference / Post-Processing
    # -------------------------------------------------------------------------
    knn_k = 100  # Number of neighbors to retrieve for re-ranking
