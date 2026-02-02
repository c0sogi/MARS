import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import provided library components
from library.utils import set_seed
from library.data_loader import load_data, IcebergDataset, get_transforms
from library.model import StabilizedSECNN
from library.trainer import train_one_epoch, validate


def run_demo():
    print("Initializing Demo...")
    # 1. Setup
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 2. Data Loading
    print("\n--- Loading Data ---")
    # load_data handles caching and metadata internally
    data = load_data(load_cached_data=True)

    # Verify data keys and shapes
    required_keys = ["X_train", "angle_train", "y_train", "X_test", "ids_test"]
    for key in required_keys:
        if key not in data:
            raise KeyError(f"Missing key in loaded data: {key}")

    print(f"Train images shape: {data['X_train'].shape}")
    print(f"Train angles shape: {data['angle_train'].shape}")
    print(f"Train labels shape: {data['y_train'].shape}")

    # Assertions to ensure data integrity
    assert len(data["X_train"]) == len(data["angle_train"]) == len(data["y_train"])
    assert data["X_train"].shape[1:] == (75, 75, 3)  # H, W, C

    # 3. Prepare Subset for Speed (Demo Mode)
    # We use a small batch size and a small subset of data to keep runtime minimal
    BATCH_SIZE = 8
    SUBSET_SIZE = 32

    X_demo = data["X_train"][:SUBSET_SIZE]
    angle_demo = data["angle_train"][:SUBSET_SIZE]
    y_demo = data["y_train"][:SUBSET_SIZE]

    # Create Dataset and DataLoader
    train_dataset = IcebergDataset(
        X_demo, angle_demo, y_demo, transform=get_transforms("train")
    )

    # Create a validation set (using same subset for demo purposes)
    val_dataset = IcebergDataset(
        X_demo, angle_demo, y_demo, transform=get_transforms("test")
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"\nCreated data loaders with subset size: {SUBSET_SIZE}")

    # 4. Model Initialization and Verification
    print("\n--- Initializing Model ---")
    model = StabilizedSECNN().to(device)

    # Verify Forward Pass logic
    # Create dummy input: Batch=2, Channels=3, H=75, W=75
    dummy_img = torch.randn(2, 3, 75, 75).to(device)
    dummy_angle = torch.tensor([35.0, 40.0]).to(device)

    with torch.no_grad():
        dummy_out = model(dummy_img, dummy_angle)

    print(f"Model output shape: {dummy_out.shape}")
    assert dummy_out.shape == (
        2,
        1,
    ), f"Expected output shape (2, 1), got {dummy_out.shape}"
    print("Model forward pass verification successful.")

    # 5. Training Loop Demonstration
    print("\n--- Starting Training Demo ---")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    EPOCHS = 2
    for epoch in range(EPOCHS):
        # Use library functions for training and validation
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

        # Verify loss is valid
        assert np.isfinite(train_loss), "Training loss is NaN or Infinite"
        assert np.isfinite(val_loss), "Validation loss is NaN or Infinite"

    # 6. Inference and Submission
    print("\n--- Generating Submission ---")
    # Use a small subset of test data for demo
    test_subset_size = 20
    X_test_sub = data["X_test"][:test_subset_size]
    angle_test_sub = data["angle_test"][:test_subset_size]
    ids_test_sub = data["ids_test"][:test_subset_size]

    test_dataset = IcebergDataset(
        X_test_sub, angle_test_sub, transform=get_transforms("test")
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model.eval()
    preds = []
    with torch.no_grad():
        for imgs, angles in test_loader:
            imgs = imgs.to(device)
            angles = angles.to(device)

            logits = model(imgs, angles)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds.extend(probs)

    preds = np.array(preds)

    # Create submission DataFrame
    submission = pd.DataFrame({"id": ids_test_sub, "is_iceberg": preds})

    # Verify submission format
    print("Submission Head:")
    print(submission.head())

    output_path = "./working/demo_submission.csv"
    submission.to_csv(output_path, index=False)
    print(f"\nSubmission saved to {output_path}")

    # Final check
    assert os.path.exists(output_path), "Submission file was not created."
    print("Demo completed successfully.")


if __name__ == "__main__":
    run_demo()
