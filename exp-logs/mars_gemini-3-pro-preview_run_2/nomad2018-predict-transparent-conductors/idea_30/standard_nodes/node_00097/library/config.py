import os
import torch


class Config:
    # --- Random Seed ---
    seed = 42

    # --- Data Paths ---
    input_dir = "./input"
    metadata_dir = "./metadata"
    train_metadata_path = os.path.join(metadata_dir, "train_metadata.csv")
    val_metadata_path = os.path.join(metadata_dir, "val_metadata.csv")
    test_metadata_path = os.path.join(metadata_dir, "test_metadata.csv")

    # --- Working Directories ---
    # Specific to this idea run
    working_dir = "./working/idea_30"
    cache_dir = os.path.join(working_dir, "cache")
    checkpoint_dir = os.path.join(working_dir, "checkpoints")
    submission_dir = os.path.join(working_dir, "submission")

    # Ensure directories exist
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(submission_dir, exist_ok=True)

    # --- Data Processing Hyperparameters ---
    cutoff_radius = 5.0  # Angstroms, strictly set as per description
    max_neighbors = 50  # Reasonable limit for radius graph to avoid OOM

    # --- Model Hyperparameters ---
    num_rbf_bins = 60  # Static Gaussian RBF filter bins
    hidden_channels = 128  # Embedding dimension
    num_interaction_blocks = 4  # Number of interaction blocks
    dropout = 0.1  # Dropout rate

    # --- Training Hyperparameters ---
    batch_size = 48  # As specified
    learning_rate = 1e-3  # Standard starting point for AdamW
    weight_decay = 1e-4  # As specified
    num_epochs = 150  # Maximum number of epochs
    patience = 20  # Early stopping patience

    # --- Device ---
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Targets ---
    target_cols = ["formation_energy_ev_natom", "bandgap_energy_ev"]

    def __init__(self):
        # Print configuration on initialization
        print(f"Config initialized.")
        print(f"  Device: {self.device}")
        print(f"  Working Directory: {self.working_dir}")
        print(f"  Batch Size: {self.batch_size}")
        print(f"  Hidden Channels: {self.hidden_channels}")
        print(f"  Interaction Blocks: {self.num_interaction_blocks}")
