import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

from library.utils import set_seed, extract_image_id
from library.data import BraTSDataset, generate_dataset_arrays, IMG_SIZE, TOTAL_CHANNELS
from library.model import RFMHDNetwork
from library.train import train_one_epoch, validate


def main():
    print("=== Starting Demonstration ===")

    # 1. Setup and Utils Verification
    print("\n[1] Setting up environment and verifying utils...")
    set_seed(42)

    # Verify ID extraction logic
    test_filename = "Image-102.dcm"
    extracted_id = extract_image_id(test_filename)
    assert extracted_id == 102, f"Expected 102, got {extracted_id}"
    print(f"Utils check passed: {test_filename} -> {extracted_id}")

    # 2. Data Processing (Subset)
    print("\n[2] Loading metadata and processing a data subset...")
    metadata_path = "./metadata/train.parquet"
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata not found at {metadata_path}")

    df = pd.read_parquet(metadata_path)

    # Select a small subset for demonstration (top 4 samples)
    # This avoids processing the entire dataset which would take too long
    subset_df = df.head(4).copy()
    print(f"Processing subset of {len(subset_df)} patients...")

    # Generate tensors using the library function
    # This reads DICOMs, normalizes, and stacks them into (128, 224, 224) tensors
    X_subset, y_subset, ids_subset = generate_dataset_arrays(
        subset_df, desc="Demo Subset"
    )

    # Verify shapes
    expected_shape = (4, TOTAL_CHANNELS, IMG_SIZE, IMG_SIZE)
    assert (
        X_subset.shape == expected_shape
    ), f"Expected X shape {expected_shape}, got {X_subset.shape}"
    assert y_subset.shape == (4,), f"Expected y shape (4,), got {y_subset.shape}"
    print(
        f"Data processed successfully. X shape: {X_subset.shape}, y shape: {y_subset.shape}"
    )

    # 3. Dataset and DataLoader Creation
    print("\n[3] Creating Dataset and DataLoaders...")
    # Create a training set from the subset
    train_dataset = BraTSDataset(
        torch.from_numpy(X_subset), torch.from_numpy(y_subset).unsqueeze(1)
    )

    # Create a validation set (reusing the same subset for demo purposes)
    val_dataset = BraTSDataset(
        torch.from_numpy(X_subset), torch.from_numpy(y_subset).unsqueeze(1)
    )

    batch_size = 2
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print(f"DataLoaders created with batch size {batch_size}.")

    # 4. Model Instantiation
    print("\n[4] Initializing RFMHDNetwork...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Initialize model (pretrained=False for speed/offline safety in demo)
    model = RFMHDNetwork(pretrained=False)
    model.to(device)

    # Verify Forward Pass
    dummy_input = torch.randn(2, TOTAL_CHANNELS, IMG_SIZE, IMG_SIZE).to(device)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (2, 1), f"Expected output shape (2, 1), got {output.shape}"
    print("Model forward pass verification successful.")

    # 5. Training Loop Demonstration
    print("\n[5] Running Training Loop (1 Epoch)...")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train for one epoch
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    print(f"Train Loss: {train_loss:.4f}")

    # Validate
    val_loss, val_auc = validate(model, val_loader, criterion, device)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation AUC:  {val_auc:.4f}")

    assert not np.isnan(train_loss), "Training loss is NaN"
    assert not np.isnan(val_loss), "Validation loss is NaN"

    # 6. Inference / Submission Logic Demonstration
    print("\n[6] Simulating Inference on Test Data...")
    model.eval()
    predictions = []
    test_ids = []

    # We use the val_loader as a proxy for test_loader here
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            predictions.extend(probs)
            # In a real test loader, the second return value would be IDs, not targets
            # We simulate IDs here
            test_ids.extend(["00000", "00001"][: len(inputs)])

    submission_df = pd.DataFrame({"BraTS21ID": test_ids, "MGMT_value": predictions})
    print("Sample predictions generated:")
    print(submission_df.head())

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
