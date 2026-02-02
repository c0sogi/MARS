import os
import torch


class Config:
    """
    Configuration for High-Capacity Unnormalized Physics-Injected Composite CNN-LSTM-FFN.
    Centralizes all hyperparameters, file paths, and feature definitions.
    """

    # --------------------------------------------------------------------------
    # Reproducibility & Environment
    # --------------------------------------------------------------------------
    seed = 42
    num_workers = 12  # Utilization of available vCPUs
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------------------------------------------------------------
    # File Paths
    # --------------------------------------------------------------------------
    # Input Metadata (Generated in ./metadata)
    train_path = "./metadata/train.csv"
    val_path = "./metadata/val.csv"
    test_path = "./metadata/test.csv"
    sample_submission_path = "./input/sample_submission.csv"

    # Working Directory & Outputs
    exp_name = "idea_optimized"
    working_dir = os.path.join("./working", exp_name)
    cache_dir = os.path.join(working_dir, "cache")
    model_path = os.path.join(working_dir, "model.pth")

    # Final Submission
    submission_dir = "./submission"
    submission_path = os.path.join(submission_dir, "submission.csv")

    # --------------------------------------------------------------------------
    # Data Configuration
    # --------------------------------------------------------------------------
    debug = False  # Set to True to use a small subset for debugging

    # Column Definitions
    id_col = "id"
    breath_id_col = "breath_id"
    time_col = "time_step"
    target_col = "pressure"

    # Feature Engineering Specification
    # The data pipeline must generate these features.
    # Includes raw controls, corrected integration (volume), dynamics (lags/diffs),
    # and physics-based interaction terms.
    features = [
        "time_step",
        "u_in",
        "u_out",
        "R",
        "C",
        "volume",  # Corrected integral of u_in * dt
        "u_in_lag1",
        "u_in_lag2",
        "u_in_lag3",
        "u_in_lag4",
        "u_in_diff1",
        "u_in_diff2",
        "u_in_R",  # Interaction: u_in * R (Pressure drop proxy)
        "volume_C",  # Interaction: volume / C (Elastic pressure proxy)
    ]

    # Physics Injection Features
    # These specific features are concatenated to the residual stream
    # at the input of each Composite Block to enforce physical constraints.
    physics_features = ["R", "C", "u_in_R", "volume_C"]

    # --------------------------------------------------------------------------
    # Model Architecture
    # --------------------------------------------------------------------------
    input_dim = len(features)
    hidden_dim = 512
    stem_kernel_sizes = [3, 5, 7]  # Multi-scale Inception-style stem
    num_blocks = 4
    aux_head_block_idx = 1  # Attach auxiliary head after the 2nd block (0-indexed)
    dropout = 0.1

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    epochs = 30
    batch_size = 512

    # Optimization (AdamW + OneCycleLR)
    learning_rate = 1e-3
    weight_decay = 1e-2
    pct_start = 0.3
    div_factor = 25.0
    final_div_factor = 10000.0

    # Loss Function
    aux_loss_weight = 0.3  # Weight for auxiliary head loss in the composite loss

    @classmethod
    def setup(cls):
        """Creates necessary output directories."""
        os.makedirs(cls.working_dir, exist_ok=True)
        os.makedirs(cls.cache_dir, exist_ok=True)
        os.makedirs(cls.submission_dir, exist_ok=True)


# Initialize directories on import
Config.setup()
