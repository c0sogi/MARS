import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import provided library components
from library.utils import seed_everything, get_device
from library.data import get_dataset_arrays, BraTSDataset
from library.model import SiameseRSFNet
from library.train import train_one_epoch, validate, predict


def run_demo():
    print(">>> Starting Library Demonstration...")

    # 1. Setup
    seed_everything(42)
    device = get_device()
    print(f"Device detected: {device}")

    # Define paths
    original_meta_path = "./metadata/train.parquet"
    demo_meta_path = "./working/demo_train.parquet"
    cache_name = "demo_run"

    # 2. Create a tiny dataset for demonstration (Speed Optimization)
    # We load the full metadata, take the first 4 rows, and save it as a new parquet file.
    # This prevents the data loader from processing hundreds of patients.
    print(f"Creating subset metadata at {demo_meta_path}...")
    df = pd.read_parquet(original_meta_path)
    df_subset = df.head(4).copy()
    df_subset.to_parquet(demo_meta_path)

    # 3. Data Processing (library.data)
    print("Processing data using library.data.get_dataset_arrays...")
    # This function handles DICOM loading, resizing, normalization, and splitting into Even/Odd streams
    # It caches results in ./working/idea_42/
    X_even, X_odd, y, ids = get_dataset_arrays(
        metadata_path=demo_meta_path,
        cache_name=cache_name,
        load_cached_data=False,  # Force processing for this demo
        input_dir="./input",
    )

    # Verification of Data Shapes
    # Expected: (N_samples, Channels=64, Height=224, Width=224)
    print(f"Data Shapes -> Even: {X_even.shape}, Odd: {X_odd.shape}, Labels: {y.shape}")
    assert len(X_even) == 4, "Expected 4 samples in the subset."
    assert X_even.shape == X_odd.shape, "Even and Odd streams must have same shape."
    assert X_even.shape[1] == 64, "Expected 64 channels (4 modalities * 16 slices)."
    assert (
        X_even.shape[2] == 224 and X_even.shape[3] == 224
    ), "Expected 224x224 image size."

    # 4. Dataset and DataLoader
    print("Initializing BraTSDataset and DataLoader...")
    dataset = BraTSDataset(X_even, X_odd, y)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    # Verification of DataLoader
    batch_xe, batch_xo, batch_y = next(iter(loader))
    assert batch_xe.shape == (2, 64, 224, 224), "Incorrect batch shape for X_even"
    assert batch_y.shape == (
        2,
        1,
    ), "Incorrect batch shape for labels (should be unsqueezed)"

    # 5. Model Initialization (library.model)
    print("Initializing SiameseRSFNet...")
    model = SiameseRSFNet(
        backbone_name="efficientnet_b0", pretrained=False
    )  # Pretrained=False for speed
    model.to(device)

    # Verification of Forward Pass
    print("Verifying model forward pass...")
    model.eval()
    with torch.no_grad():
        batch_xe = batch_xe.to(device)
        batch_xo = batch_xo.to(device)
        logits = model(batch_xe, batch_xo)

    print(f"Logits shape: {logits.shape}")
    assert logits.shape == (2, 1), "Model output should be (Batch_Size, 1)"

    # 6. Training Loop Demonstration (library.train)
    print("Running training loop (1 epoch)...")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # train_one_epoch returns (avg_loss, auc)
    train_loss, train_auc = train_one_epoch(model, loader, criterion, optimizer, device)
    print(f"Train Result -> Loss: {train_loss:.4f}, AUC: {train_auc:.4f}")

    assert not np.isnan(train_loss), "Training loss is NaN"
    assert 0.0 <= train_auc <= 1.0, "AUC must be between 0 and 1"

    # 7. Validation Demonstration
    print("Running validation...")
    val_loss, val_auc = validate(model, loader, criterion, device)
    print(f"Val Result   -> Loss: {val_loss:.4f}, AUC: {val_auc:.4f}")

    # 8. Prediction Demonstration
    print("Running prediction...")
    # Create a test dataset (no labels) using the same data for simplicity
    test_dataset = BraTSDataset(X_even, X_odd, y=None)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    preds = predict(model, test_loader, device)
    print(f"Predictions shape: {preds.shape}")
    print(f"Sample predictions: {preds}")

    assert len(preds) == 4, "Expected 4 predictions for 4 samples."
    assert np.all(
        (preds >= 0) & (preds <= 1)
    ), "Predictions must be probabilities [0, 1]"

    print("\n>>> Demonstration completed successfully.")


if __name__ == "__main__":
    run_demo()
