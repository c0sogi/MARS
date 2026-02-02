import os
import torch


class Config:
    # Random Seed for reproducibility
    seed = 42

    # Compute environment
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 4

    # Directories
    input_dir = "./input"
    metadata_dir = "./metadata"
    working_dir = "./working/idea_32"

    # Input Files (using Parquet metadata as requested)
    train_file = os.path.join(metadata_dir, "train.parquet")
    val_file = os.path.join(metadata_dir, "val.parquet")
    test_file = os.path.join(metadata_dir, "test.parquet")
    sample_submission_file = os.path.join(input_dir, "sample_submission.csv")

    # Output Files
    model_save_path = os.path.join(working_dir, "best_model.pth")
    submission_path = os.path.join(working_dir, "submission.csv")

    # Data Configuration
    seq_len = 107
    pred_len = 68
    # Explicitly targeting the 3 scored columns
    target_cols = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    # Model Architecture: Wide-Stream Residual BiGRU
    embed_dim = 128  # High-Dimensional Embeddings
    hidden_dim = 384  # Balanced Capacity (Cite solution_lesson_node_00081)
    n_layers = 6  # Backbone Depth
    dropout = 0.1  # Inter-Layer Dropout

    # Training Hyperparameters
    epochs = 20
    batch_size = 64  # Sized for A100 GPU
    lr = 1e-3  # Standard AdamW learning rate
    weight_decay = 1e-4  # Low weight decay to preserve recurrent signals

    def __init__(self):
        # Ensure the working directory exists upon instantiation
        os.makedirs(self.working_dir, exist_ok=True)
