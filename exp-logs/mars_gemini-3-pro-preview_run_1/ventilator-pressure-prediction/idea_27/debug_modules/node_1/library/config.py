import os
import torch


class Config:
    """
    Configuration for the Wide-Projected Deeply-Supervised Physics-Identity Network.

    Strategy:
    - Wide Projection: Decouples feature extraction (stem_dim=512) from latent modeling (model_dim=1024).
    - Deep Supervision: Auxiliary head on intermediate block with weight 0.3.
    - Physics-Identity: Strict identity residuals and physics-informed feature injection.
    - Extended Optimization: 35 epochs with OneCycleLR and strict gradient clipping.
    """

    # ==============================
    # General Configuration
    # ==============================
    seed = 42
    debug = False  # Set to True to run on a small subset for debugging
    num_workers = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ==============================
    # Directories & Paths
    # ==============================
    # Input (Read-Only)
    input_dir = "./input"
    metadata_dir = "./metadata"

    # Working (Read/Write) - Specific to this idea
    working_dir = "./working/idea_27"
    submission_dir = "./submission"

    # Data Files
    train_file = os.path.join(metadata_dir, "train.csv")
    val_file = os.path.join(metadata_dir, "val.csv")
    test_file = os.path.join(metadata_dir, "test.csv")
    sample_submission_file = os.path.join(input_dir, "sample_submission.csv")

    # Cache Files (Parquet/NPY for speed)
    train_cache = os.path.join(working_dir, "train_engineered.parquet")
    val_cache = os.path.join(working_dir, "val_engineered.parquet")
    test_cache = os.path.join(working_dir, "test_engineered.parquet")

    # Scaler Cache
    scaler_center_path = os.path.join(working_dir, "scaler_center.npy")
    scaler_scale_path = os.path.join(working_dir, "scaler_scale.npy")

    # Model Checkpoint
    model_path = os.path.join(working_dir, "model.pth")
    output_submission_path = os.path.join(submission_dir, "submission.csv")

    # ==============================
    # Model Architecture
    # ==============================
    # Stem: Mixed Multi-Scale Initialization
    stem_dim = 512

    # Backbone: Wide-State Identity Blocks
    model_dim = 1024

    # LSTM: Bi-Directional (512 * 2 = 1024 matches model_dim for identity mapping)
    lstm_hidden = 512

    # Structure
    num_blocks = 4
    expansion_factor = 2  # FFN expansion (1024 -> 2048 -> 1024)
    dropout = 0.1

    # Deep Supervision
    aux_weight = 0.3  # Weight for auxiliary loss (L_total = L_final + 0.3 * L_aux)

    # ==============================
    # Training Hyperparameters
    # ==============================
    # Extended Optimization Horizon
    epochs = 35

    # Update Budget
    batch_size = 512

    # Optimization
    lr_max = 1e-3
    pct_start = 0.3  # OneCycleLR warm-up percentage
    weight_decay = 1e-2

    # Strict Gradient Clipping (Critical for Wide-State LSTM)
    clip_grad = 1.0

    # ==============================
    # Data & Feature Engineering
    # ==============================
    target_col = "pressure"
    id_col = "id"
    breath_id_col = "breath_id"

    # Sequence Length
    seq_len = 80

    # Feature Flags
    use_lags = True
    lag_steps = [1, 2, 3, 4]
    use_diffs = True  # First and Second differences of u_in

    # Physics Features
    # We explicitly calculate Volume (integral of u_in) and interactions
    # These are handled in the dataset processing pipeline

    @classmethod
    def setup(cls):
        """Ensures necessary working directories exist."""
        os.makedirs(cls.working_dir, exist_ok=True)
        os.makedirs(cls.submission_dir, exist_ok=True)

    @classmethod
    def print_config(cls):
        """Prints the current configuration."""
        print(f"=== Configuration ===")
        print(f"Device: {cls.device}")
        print(f"Working Dir: {cls.working_dir}")
        print(
            f"Architecture: Stem={cls.stem_dim}, Model={cls.model_dim}, LSTM={cls.lstm_hidden}x2"
        )
        print(
            f"Training: Epochs={cls.epochs}, Batch={cls.batch_size}, Clip={cls.clip_grad}"
        )
        print(f"Deep Supervision Weight: {cls.aux_weight}")
        print(f"=====================")
