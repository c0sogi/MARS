import os
import torch
import random
import numpy as np


class Config:
    """
    Configuration class for the Ventilator Pressure Prediction task.
    Implements the Physics-Residual Multi-Scale CNN-LSTM strategy settings.
    """

    # -------------------------------------------------------------------------
    # General Configuration
    # -------------------------------------------------------------------------
    seed = 42
    debug = False  # Set to True to use a smaller subset of data for debugging
    exp_name = "idea_4"  # Experiment identifier

    # -------------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------------
    # Input data paths (using generated metadata splits)
    train_file = "./metadata/train.csv"
    val_file = "./metadata/val.csv"
    test_file = "./metadata/test.csv"
    sample_submission_file = "./input/sample_submission.csv"

    # Working directory for artifacts
    working_dir = os.path.join("./working", exp_name)
    os.makedirs(working_dir, exist_ok=True)

    # Output file paths
    model_path = os.path.join(working_dir, "model.pth")
    submission_path = os.path.join(working_dir, "submission.csv")

    # Cache directory for processed datasets (numpy/parquet)
    cache_dir = working_dir

    # -------------------------------------------------------------------------
    # Data Configuration
    # -------------------------------------------------------------------------
    # Sequence length is fixed by the breath duration in the dataset
    seq_len = 80

    # Column definitions
    id_col = "id"
    breath_id_col = "breath_id"
    time_col = "time_step"
    target_col = "pressure"

    # Features to be used by the model
    # Includes raw control inputs, lung attributes, and physics-informed engineering
    feature_cols = [
        "time_step",
        "u_in",
        "u_out",
        "R",
        "C",
        "cumulative_volume",  # Integral of u_in over time (Volume)
        "flow_interaction",  # u_in * R (Resistive Pressure component)
        "vol_interaction",  # cumulative_volume / C (Elastic Pressure component)
    ]

    # Input dimension for the model stem
    input_dim = len(feature_cols)

    # -------------------------------------------------------------------------
    # Model Architecture (Physics-Residual Multi-Scale CNN-LSTM)
    # -------------------------------------------------------------------------
    # Multi-Scale 1D CNN Stem
    cnn_filters = 64
    cnn_kernel_sizes = [3, 5, 7]  # Inception-style parallel convolutions
    cnn_dropout = 0.1

    # The LSTM input size is the concatenated output of the multi-scale CNNs
    lstm_input_size = cnn_filters * len(cnn_kernel_sizes)

    # Residual Bi-LSTM Backbone
    lstm_hidden_size = 512
    lstm_layers = 4
    bidirectional = True
    lstm_dropout = 0.1  # Dropout between LSTM layers

    # -------------------------------------------------------------------------
    # Training Configuration
    # -------------------------------------------------------------------------
    epochs = 100
    train_batch_size = 256  # A100 GPU allows for larger batch sizes
    val_batch_size = 512

    # Optimization (AdamW)
    learning_rate = 1e-3
    weight_decay = 1e-2
    max_grad_norm = 1000.0  # Gradient clipping

    # Scheduler (OneCycleLR)
    pct_start = 0.1
    div_factor = 25.0
    final_div_factor = 1e4

    # Early Stopping
    patience = 15

    # Hardware
    num_workers = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------
    @staticmethod
    def set_seed():
        """
        Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
        """
        seed = Config.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)

    @staticmethod
    def print_config():
        """
        Prints the current configuration setup.
        """
        print(f"=== Configuration ({Config.exp_name}) ===")
        print(f"  Device:        {Config.device}")
        print(f"  Debug Mode:    {Config.debug}")
        print(f"  Model:         Physics-Residual Multi-Scale CNN-LSTM")
        print(f"  Input Dim:     {Config.input_dim}")
        print(f"  Features:      {Config.feature_cols}")
        print(f"  CNN Kernels:   {Config.cnn_kernel_sizes}")
        print(
            f"  LSTM:          {Config.lstm_layers} layers x {Config.lstm_hidden_size} hidden (BiDir={Config.bidirectional})"
        )
        print(f"  Batch Size:    {Config.train_batch_size}")
        print(f"  Max Epochs:    {Config.epochs}")
        print(f"==========================================")
