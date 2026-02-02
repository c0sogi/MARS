import os


class Config:
    # ==============================
    # General Configuration
    # ==============================
    seed = 42
    debug = False  # Set to True to run on a small subset for debugging
    num_workers = 4  # Number of dataloader workers

    # ==============================
    # Paths
    # ==============================
    input_dir = "./input"
    metadata_dir = "./metadata"
    working_dir = "./working/idea_20"

    # Ensure working directory exists
    os.makedirs(working_dir, exist_ok=True)

    # Data Paths
    train_path = os.path.join(metadata_dir, "train.parquet")
    val_path = os.path.join(metadata_dir, "val.parquet")
    test_path = os.path.join(metadata_dir, "test.parquet")
    sample_submission_path = os.path.join(input_dir, "sample_submission.csv")

    # Cache Paths (for deterministic data processing)
    train_cache_path = os.path.join(working_dir, "train_data.npz")
    val_cache_path = os.path.join(working_dir, "val_data.npz")
    test_cache_path = os.path.join(working_dir, "test_data.npz")

    # Output Paths
    model_save_path = os.path.join(working_dir, "best_model.pth")
    submission_path = os.path.join(working_dir, "submission.csv")

    # ==============================
    # Data Parameters
    # ==============================
    seq_len = 107
    seq_scored = 68

    # Input Channels: 4 (Seq) + 3 (Struct) + 7 (Loop) = 14
    input_channels = 14

    # Targets
    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    num_targets = len(target_cols)

    # Token Mappings for One-Hot Encoding
    token2int_seq = {"A": 0, "G": 1, "C": 2, "U": 3}
    token2int_struct = {".": 0, "(": 1, ")": 2}
    token2int_loop = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

    # ==============================
    # Model Hyperparameters (CGSR-BiGRU)
    # ==============================
    # Convolutional Stem
    conv_kernel_size = 3
    conv_filters = 256

    # Recurrent Backbone
    hidden_dim = 384
    n_layers = 3  # Number of BiGRU + Interaction Blocks
    dropout = 0.1

    # ==============================
    # Training Hyperparameters
    # ==============================
    batch_size = 32
    learning_rate = 1e-3
    epochs = 20

    # Optimization
    max_grad_norm = 1.0  # Gradient Clipping
    weight_decay = 1e-4

    # Scheduler (Cosine Annealing)
    T_max = epochs
    min_lr = 1e-6
