import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import provided library modules
from library.utils import set_seed, get_device
from library.dataset import process_data, BraTSDataset
from library.model import MGMTNet
from library.train import train_one_epoch, validate


def run_demo():
    print("----------------------------------------------------------------")
    print("Starting Glioblastoma Classification Library Demo")
    print("----------------------------------------------------------------")

    # 1. Setup
    set_seed(42)
    device = get_device()
    print(f"Device selected: {device}")

    # Define paths
    metadata_path = "./metadata/train.parquet"
    cache_key = "demo_execution"

    # 2. Data Processing & Dataset Demonstration
    print("\n[Dataset] Loading metadata and creating a mini-dataset...")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    # Load full metadata
    df = pd.read_parquet(metadata_path)

    # Select only 2 samples for the demo to ensure speed
    mini_df = df.head(2).copy()
    print(f"Selected {len(mini_df)} samples for demonstration.")

    # Process data (Load DICOMs -> Normalize -> Resize -> Stack)
    # We force load_cached_data=False to demonstrate the processing logic
    print("Processing MRI volumes (this involves reading DICOM files)...")
    X, y, ids = process_data(mini_df, cache_key, load_cached_data=False)

    # Verify Data Shapes
    # Expected X shape: (N, Channels, H, W) -> (2, 128, 256, 256)
    # Channels = 4 modalities * 32 slices = 128
    assert X.shape == (
        2,
        128,
        256,
        256,
    ), f"Expected X shape (2, 128, 256, 256), got {X.shape}"
    assert y.shape == (2,), f"Expected y shape (2,), got {y.shape}"
    assert len(ids) == 2, "IDs length mismatch"

    # Instantiate Dataset
    dataset = BraTSDataset(X, y, ids)

    # Verify Dataset __getitem__
    sample_x, sample_y = dataset[0]
    print(f"Dataset sample shape: {sample_x.shape}, Target: {sample_y}")
    assert sample_x.shape == (128, 256, 256)
    assert isinstance(sample_y.item(), float)

    # Create DataLoader
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    # 3. Model Demonstration
    print("\n[Model] Instantiating MGMTNet...")
    model = MGMTNet().to(device)

    # Verify Forward Pass with Dummy Data
    print("Verifying forward pass with dummy input...")
    dummy_input = torch.randn(2, 128, 256, 256).to(device)
    with torch.no_grad():
        dummy_output = model(dummy_input)

    print(f"Output shape: {dummy_output.shape}")
    assert dummy_output.shape == (
        2,
        1,
    ), f"Expected output shape (2, 1), got {dummy_output.shape}"

    # 4. Training Loop Demonstration
    print("\n[Training] Simulating one epoch of training...")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Run one training epoch
    train_loss = train_one_epoch(model, loader, criterion, optimizer, device)
    print(f"Training Loss: {train_loss:.4f}")
    assert train_loss >= 0, "Training loss should be non-negative"

    # Run validation
    print("Simulating validation...")
    val_loss, val_auc = validate(model, loader, criterion, device)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation AUC: {val_auc:.4f}")

    assert val_loss >= 0, "Validation loss should be non-negative"
    assert 0.0 <= val_auc <= 1.0, "AUC should be between 0 and 1"

    # 5. Inference Example
    print("\n[Inference] Generating predictions...")
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = torch.sigmoid(logits)
            predictions.extend(probs.cpu().numpy().flatten())

    print(f"Predictions: {predictions}")
    assert len(predictions) == 2
    assert all(
        0.0 <= p <= 1.0 for p in predictions
    ), "Predictions must be probabilities"

    print("\n----------------------------------------------------------------")
    print("Demo Completed Successfully")
    print("----------------------------------------------------------------")


if __name__ == "__main__":
    run_demo()
