import os
import torch
import pandas as pd
import numpy as np
import random
from torch.utils.data import DataLoader

# Import pre-defined classes and functions
from library.data_loader import GnssWindowedDataset
from library.model import TemporalConvNet, train_model, generate_submission


def set_seed(seed):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_training(
    input_dir="./input",
    metadata_dir="./metadata",
    working_dir="./working",
    submission_dir="./submission",
    config=None,
):
    """
    Orchestrates the training pipeline: data loading, model training, and submission generation.

    Args:
        input_dir (str): Path to the read-only input directory.
        metadata_dir (str): Path to the directory containing metadata CSVs.
        working_dir (str): Path to the directory for saving model artifacts and cache.
        submission_dir (str): Path to the directory for saving the submission file.
        config (dict, optional): Hyperparameters and configuration. Defaults are used if None.

    Returns:
        dict: Training history containing train and validation loss logs.
    """
    # Default configuration
    if config is None:
        config = {
            "window_size": 64,
            "batch_size": 256,
            "lr": 1e-3,
            "epochs": 15,
            "patience": 5,
            "hidden_dim": 128,
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            "seed": 42,
        }

    # Set reproducibility
    set_seed(config["seed"])

    # Ensure output directories exist
    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(submission_dir, exist_ok=True)

    print(f"Configuration: {config}")

    # --- 1. Load Metadata ---
    train_meta_path = os.path.join(metadata_dir, "train_metadata.csv")
    val_meta_path = os.path.join(metadata_dir, "val_metadata.csv")
    test_meta_path = os.path.join(metadata_dir, "test_metadata.csv")

    if not os.path.exists(train_meta_path) or not os.path.exists(val_meta_path):
        raise FileNotFoundError(
            "Metadata files not found. Please ensure metadata generation is complete."
        )

    df_train = pd.read_csv(train_meta_path)
    df_val = pd.read_csv(val_meta_path)

    print(
        f"Metadata loaded. Training samples: {len(df_train)}, Validation samples: {len(df_val)}"
    )

    # --- 2. Initialize Datasets and Loaders ---
    # Note: GnssWindowedDataset handles caching internally via preprocess_drive
    # We pass the working_dir implicitly via the CACHE_DIR constant in data_loader,
    # but here we rely on the library's default or we could modify the library if needed.
    # Assuming library defaults are acceptable or pointing to ./working/idea_1/

    print("Initializing Training Dataset...")
    # Train dataset fits the scaler
    train_dataset = GnssWindowedDataset(
        metadata_df=df_train,
        input_dir=input_dir,
        window_size=config["window_size"],
        mode="train",
        scaler=None,
    )

    print("Initializing Validation Dataset...")
    # Validation dataset uses the scaler fitted on training data
    val_dataset = GnssWindowedDataset(
        metadata_df=df_val,
        input_dir=input_dir,
        window_size=config["window_size"],
        mode="train",  # 'train' mode implies ground truth is available
        scaler=train_dataset.scaler,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # --- 3. Initialize Model ---
    # Input channels: WlsAlt, Cn0DbHz, SvElevationDegrees, SatCount, RawPseudorangeUncertaintyMeters (5 features)
    # Output dim: dLat, dLon (2 targets)
    model = TemporalConvNet(
        input_channels=5,
        window_size=config["window_size"],
        hidden_dim=config["hidden_dim"],
        output_dim=2,
    )

    # --- 4. Train Model ---
    trained_model, history = train_model(
        model=model, train_loader=train_loader, val_loader=val_loader, config=config
    )

    # --- 5. Save Artifacts ---
    model_save_path = os.path.join(working_dir, "model_weights.pth")
    torch.save(trained_model.state_dict(), model_save_path)
    print(f"Best model weights saved to {model_save_path}")

    # Print final metrics
    if history["val_loss"]:
        print(f"Final Validation MAE: {history['val_loss'][-1]}")
        print(f"Best Validation MAE: {min(history['val_loss'])}")

    return history
