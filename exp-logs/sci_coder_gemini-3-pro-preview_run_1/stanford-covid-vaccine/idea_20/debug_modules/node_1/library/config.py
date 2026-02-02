import os
import torch


class Config:
    # ==============================
    # General Configuration
    # ==============================
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 4  # Number of dataloader workers

    # ==============================
    # Paths & Directories
    # ==============================
    # Input directories (Read-Only)
    input_dir = "./input"
    metadata_dir = "./metadata"

    # Metadata file paths
    train_file = os.path.join(metadata_dir, "train.parquet")
    val_file = os.path.join(metadata_dir, "val.parquet")
    test_file = os.path.join(metadata_dir, "test.parquet")

    # Output directories (Write Allowed)
    working_dir = "./working/idea_20"
    model_save_path = os.path.join(working_dir, "best_model.pth")
    predictions_dir = os.path.join(working_dir, "predictions")
    submission_path = "./submission/submission.csv"

    # Create necessary writable directories
    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)

    # ==============================
    # Data Specifications
    # ==============================
    seq_len = 107
    pred_len = 68

    # The 3 columns that are actually scored in the competition
    scored_targets = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    n_targets = len(scored_targets)

    # Mappings for Atomic Sequence (A, G, U, C)
    vocab_map = {"A": 0, "C": 1, "G": 2, "U": 3}
    vocab_size = len(vocab_map)

    # Mappings for Predicted Loop Type
    # S: Stem, M: Multiloop, I: Internal loop, B: Bulge, H: Hairpin, E: Dangling End, X: External
    loop_type_map = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}
    loop_vocab_size = len(loop_type_map)

    # ==============================
    # Model Architecture
    # ==============================
    # Input Embeddings
    seq_embed_dim = 64  # Embedding size for nucleotide identity
    loop_embed_dim = 32  # Embedding size for loop type
    distance_dim = 32  # Dimension for sinusoidal pair distance encoding

    # Wide-Stream BiGRU Backbone
    hidden_dim = 384  # Width of the residual stream (Wide-Stream)
    n_layers = 6  # Number of residual blocks
    dropout = 0.1  # Dropout rate

    # The model input dimension is the sum of feature embeddings
    input_dim = seq_embed_dim + loop_embed_dim + distance_dim

    # ==============================
    # Training Hyperparameters
    # ==============================
    batch_size = 32  # Adjust based on GPU memory (A100 40GB can handle larger)
    lr = 1e-3  # Initial learning rate
    weight_decay = 1e-4  # Weight decay for AdamW
    max_grad_norm = 1.0  # Gradient clipping

    epochs = 100  # Maximum training epochs (utilize 24h limit)
    es_patience = 15  # Early stopping patience (epochs without improvement)

    # Scheduler: ReduceLROnPlateau
    scheduler_mode = "min"  # Minimize MCRMSE
    scheduler_factor = 0.5  # Decay factor
    scheduler_patience = 5  # Patience before decay
    min_lr = 1e-6  # Minimum learning rate

    # ==============================
    # Debugging
    # ==============================
    debug = False  # If True, runs on a small subset of data
    debug_subset_size = 100
