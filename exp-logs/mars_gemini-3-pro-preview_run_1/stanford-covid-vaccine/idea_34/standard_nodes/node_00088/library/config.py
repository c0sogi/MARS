import os
import torch


class Config:
    """
    Configuration class for the Bond-Aware Wide-Stream Residual BiGRU model.
    Centralizes all hyperparameters, file paths, and data settings.
    """

    # --------------------------
    # General Settings
    # --------------------------
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 2  # Number of dataloader workers
    debug = False  # Set to True to run on a small subset for debugging

    # --------------------------
    # File Paths
    # --------------------------
    # Input directories (Read-Only)
    input_dir = "./input"
    metadata_dir = "./metadata"

    # Working directory (Write Allowed)
    working_dir = "./working/idea_34"

    # Data Files
    train_file = os.path.join(metadata_dir, "train.parquet")
    val_file = os.path.join(metadata_dir, "val.parquet")
    test_file = os.path.join(metadata_dir, "test.parquet")
    sample_submission_file = os.path.join(input_dir, "sample_submission.csv")

    # Cache Files (for deterministic data processing)
    train_cache = os.path.join(working_dir, "train_data.pt")
    val_cache = os.path.join(working_dir, "val_data.pt")
    test_cache = os.path.join(working_dir, "test_data.pt")

    # Output Files
    model_path = os.path.join(working_dir, "best_model.pth")
    submission_path = os.path.join(working_dir, "submission.csv")

    # --------------------------
    # Data Parameters
    # --------------------------
    seq_len = 107
    pred_len = 68

    # Target columns to train on (and predict)
    # Note: We filter out deg_pH10 and deg_50C for training as per strategy
    target_cols = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    num_targets = len(target_cols)

    # Vocabularies
    # 1. Sequence (Atomic)
    token2id = {"A": 0, "G": 1, "C": 2, "U": 3}
    vocab_size_seq = len(token2id)

    # 2. Predicted Loop Type
    loop2id = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}
    vocab_size_loop = len(loop2id)

    # 3. Bond Type (Soft Feature)
    # Pairs: A-U, U-A, G-C, C-G, G-U, U-G, Mismatch, Unpaired
    bond2id = {
        "A-U": 0,
        "U-A": 1,
        "G-C": 2,
        "C-G": 3,
        "G-U": 4,
        "U-G": 5,
        "Mismatch": 6,
        "Unpaired": 7,
    }
    vocab_size_bond = len(bond2id)

    # --------------------------
    # Model Architecture
    # --------------------------
    # Embeddings
    emb_dim = 128

    # Recurrent Backbone
    # Wide-Stream: Hidden dim maintained throughout
    hidden_dim = 384
    n_layers = 6
    dropout = 0.1

    # --------------------------
    # Training Hyperparameters
    # --------------------------
    batch_size = 32
    epochs = 20

    # Optimizer
    lr = 1e-3
    weight_decay = 1e-4

    # Scheduler
    T_max = epochs  # For CosineAnnealingLR
    min_lr = 1e-6

    def __init__(self):
        """
        Initialize configuration.
        Ensures the working directory exists.
        """
        os.makedirs(self.working_dir, exist_ok=True)
