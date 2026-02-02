import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from library.utils import seed_everything, get_device, save_submission
from library.data_processing import feature_engineering, ForestDataset
from library.model import AsymmetricParallelNet, predict
from library.train import train_one_epoch, validate

if __name__ == "__main__":
    print("Starting demonstration script...")

    # 1. Setup
    seed_everything(42)
    device = get_device()
    print(f"Device: {device}")

    # 2. Data Preparation (Synthetic for Speed)
    # We create a small synthetic dataset to demonstrate the pipeline quickly
    # without processing the full 3M row dataset which would exceed the time limit.
    print("Generating synthetic data...")
    num_samples = 100

    # Define columns required by feature_engineering
    columns = [
        "Elevation",
        "Aspect",
        "Slope",
        "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Hillshade_9am",
        "Hillshade_Noon",
        "Hillshade_3pm",
        "Horizontal_Distance_To_Fire_Points",
    ]
    # Add binary columns (Wilderness_Area1-4, Soil_Type1-40)
    columns += [f"Wilderness_Area{i}" for i in range(1, 5)]
    columns += [f"Soil_Type{i}" for i in range(1, 41)]

    # Create random data
    data = np.random.randn(num_samples, len(columns)).astype(np.float32)
    df = pd.DataFrame(data, columns=columns)

    # Add Id and Target
    df["Id"] = np.arange(num_samples)
    df["Cover_Type"] = np.random.randint(1, 8, size=num_samples)  # Classes 1-7

    print("Applying feature engineering...")
    # Demonstrate feature_engineering function
    df_processed = feature_engineering(df)

    # Verify new columns from engineering exist
    expected_new_cols = [
        "Aspect_Sin",
        "Euclidean_Distance_To_Hydrology",
        "Mean_Distance_To_Amenities",
    ]
    for col in expected_new_cols:
        if col not in df_processed.columns:
            raise AssertionError(f"Feature engineering failed to create {col}")

    # Prepare data for model (mimicking get_dataloaders logic)
    # Drop non-feature columns
    drop_cols = ["Id", "Cover_Type"]
    feature_cols = [c for c in df_processed.columns if c not in drop_cols]

    X = df_processed[feature_cols].values.astype(np.float32)
    # Target needs to be 0-6 for CrossEntropyLoss
    y = (df_processed["Cover_Type"].values - 1).astype(np.int64)
    ids = df["Id"].values

    print(f"Processed data shape: {X.shape}")

    # Demonstrate ForestDataset
    train_dataset = ForestDataset(X, y)
    test_dataset = ForestDataset(X, None)  # No target for test

    # Create DataLoaders
    batch_size = 16
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False
    )  # Reuse for demo
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    # 3. Model Initialization
    print("Initializing model...")
    input_dim = X.shape[1]
    num_classes = 7
    model = AsymmetricParallelNet(input_dim, num_classes).to(device)

    # 4. Training Loop Demonstration
    print("Running training step...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Demonstrate train_one_epoch
    train_loss, train_acc = train_one_epoch(
        model, train_loader, optimizer, criterion, device
    )
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

    # Demonstrate validate
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # 5. Prediction and Submission
    print("Generating predictions...")
    # Demonstrate predict function
    preds = predict(model, test_loader, device)

    # Verify predictions
    if len(preds) != num_samples:
        raise AssertionError("Prediction count mismatch")
    if not np.all((preds >= 1) & (preds <= 7)):
        raise AssertionError("Predictions out of range (should be 1-7)")

    print("Saving submission...")
    submission_path = "./working/demo_submission_out.csv"
    # Demonstrate save_submission
    save_submission(preds, ids, submission_path)

    # Verify file creation
    if not os.path.exists(submission_path):
        raise AssertionError("Submission file not created")

    # Check content
    sub_df = pd.read_csv(submission_path)
    if sub_df.shape != (num_samples, 2):
        raise AssertionError("Submission file shape mismatch")

    print("Demonstration completed successfully.")
