import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Import library modules
import importlib
import library.config
import library.utils
import library.data
import library.model
import library.train
import library.inference

# Reload modules to ensure updates are picked up in persistent environments
importlib.reload(library.config)
importlib.reload(library.utils)
importlib.reload(library.data)
importlib.reload(library.model)
importlib.reload(library.train)
importlib.reload(library.inference)

from library.config import Config
from library.train import train_model
from library.inference import generate_submission
from library.data import get_dataloaders
from library.model import BiGRUModel
from library.utils import seed_everything, load_checkpoint


def main():
    # ==========================================
    # 1. Setup & Configuration Overrides
    # ==========================================
    seed_everything(Config.SEED)

    # Ensure working directory exists (Config creates it on import, but good to be safe)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print("Preparing fast baseline configuration...")

    # Load original metadata
    orig_train_path = os.path.join(Config.METADATA_DIR, "train.csv")
    orig_val_path = os.path.join(Config.METADATA_DIR, "val.csv")

    if not os.path.exists(orig_train_path) or not os.path.exists(orig_val_path):
        raise FileNotFoundError("Original metadata files not found.")

    train_df = pd.read_csv(orig_train_path)
    val_df = pd.read_csv(orig_val_path)

    # Create subsets for fast execution (Limit samples)
    # 10,000 training samples and 2,000 validation samples
    subset_train_path = os.path.join(Config.WORKING_DIR, "train_subset.csv")
    subset_val_path = os.path.join(Config.WORKING_DIR, "val_subset.csv")

    train_df.head(10000).to_csv(subset_train_path, index=False)
    val_df.head(2000).to_csv(subset_val_path, index=False)

    # Override Config attributes
    Config.TRAIN_CSV = subset_train_path
    Config.VAL_CSV = subset_val_path
    Config.EPOCHS = 5  # Reduce epochs for speed
    Config.LOAD_CACHED_DATA = True  # Use caching as requested

    # ==========================================
    # 2. Training
    # ==========================================
    print("\n=== Starting Training Phase ===")
    train_model()

    # ==========================================
    # 3. Validation & Failure Analysis
    # ==========================================
    print("\n=== Starting Validation Analysis ===")
    analyze_validation()

    # ==========================================
    # 4. Submission
    # ==========================================
    print("\n=== Generating Submission ===")
    generate_submission(load_cached_data=Config.LOAD_CACHED_DATA)

    print("\nRunfile execution complete.")


def analyze_validation():
    """
    Performs manual validation inference to compute the final metric
    and runs failure analysis (correlation of error with metadata).
    """
    device = torch.device(Config.DEVICE)

    # 1. Load Validation Data
    # We use get_dataloaders to ensure consistent preprocessing
    # We ignore train/test loaders here
    _, val_loader, _ = get_dataloaders(load_cached_data=Config.LOAD_CACHED_DATA)

    # 2. Load Model
    model = BiGRUModel()
    model.to(device)

    # Load best model
    checkpoint = load_checkpoint(model, filename="best_model.pth", device=device)
    if checkpoint is None:
        print(
            "Warning: Best model not found. Using initialized model (random weights)."
        )

    model.eval()

    # 3. Inference & Metric Calculation
    all_losses = []
    # KLDivLoss with reduction='none' returns loss per element
    criterion = nn.KLDivLoss(reduction="none")

    with torch.no_grad():
        for data, targets in val_loader:
            data = data.to(device)
            targets = targets.to(device)

            logits = model(data)
            log_probs = F.log_softmax(logits, dim=1)

            # Compute KL Divergence
            # Output shape: (Batch, Classes)
            batch_loss_map = criterion(log_probs, targets)

            # Sum over classes to get scalar KL divergence per sample
            # Shape: (Batch,)
            sample_losses = batch_loss_map.sum(dim=1)

            all_losses.extend(sample_losses.cpu().numpy())

    # Calculate Mean KL Divergence
    final_metric = np.mean(all_losses)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    # Load the subset metadata used for validation
    val_df = pd.read_csv(Config.VAL_CSV)

    # Ensure lengths match (DataLoader might drop last if configured, though val usually doesn't)
    if len(all_losses) != len(val_df):
        # If mismatch, truncate DF to match processed samples
        val_df = val_df.iloc[: len(all_losses)].copy()

    val_df["error_magnitude"] = all_losses

    # Calculate correlations
    analysis_cols = [
        "total_votes",
        "eeg_label_offset_seconds",
        "spectogram_label_offset_seconds",
    ]
    # Filter for columns that actually exist
    existing_cols = [c for c in analysis_cols if c in val_df.columns]

    if existing_cols:
        print("\nCorrelation between Error Magnitude and Metadata Features:")
        correlations = val_df[existing_cols + ["error_magnitude"]].corr()[
            "error_magnitude"
        ]
        # Drop the self-correlation
        correlations = correlations.drop("error_magnitude", errors="ignore")
        print(correlations)
    else:
        print("No metadata columns found for failure analysis.")

    return final_metric


if __name__ == "__main__":
    main()
