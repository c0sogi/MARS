import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import provided library modules
import library.config as config
import library.utils as utils
import library.data_loader as data_loader
import library.model as model_lib
import library.train as train_lib


def demo_data_loading():
    print("\n=== Demo: Data Loading ===")

    # Load data using the library function.
    # It will use cached .npy files if available, which is fast.
    print("Loading training data...")
    X, angles, y, ids = data_loader.load_data_split(
        config.TRAIN_META_PATH, config.TRAIN_JSON, "train", load_cached_data=True
    )

    # OPTIMIZATION: Create a mini-dataset (32 samples) for demonstration speed
    subset_size = 32
    X_mini = X[:subset_size]
    angles_mini = angles[:subset_size]
    y_mini = y[:subset_size]
    ids_mini = ids[:subset_size]

    # Handle NaNs in angles for the subset (simple imputation for demo)
    angles_mini = np.where(np.isnan(angles_mini), 0.0, angles_mini)

    print(f"Original dataset size: {len(X)}")
    print(f"Mini dataset size: {len(X_mini)}")

    # Create the Dataset object
    dataset = data_loader.IcebergDataset(X_mini, angles_mini, y_mini, ids_mini)

    # Create DataLoader
    batch_size = 8
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Fetch a single batch to verify shapes
    images, batch_angles, labels, batch_ids = next(iter(loader))

    print(f"Batch Images Shape: {images.shape}")  # Should be (8, 3, 75, 75)
    print(f"Batch Angles Shape: {batch_angles.shape}")  # Should be (8,)
    print(f"Batch Labels Shape: {labels.shape}")  # Should be (8, 1)

    # Verification
    assert images.shape == (batch_size, 3, 75, 75), "Incorrect image tensor shape"
    assert batch_angles.shape == (batch_size,), "Incorrect angle tensor shape"
    assert labels.shape == (batch_size, 1), "Incorrect label tensor shape"

    return loader


def demo_model_logic(device, loader):
    print("\n=== Demo: Model Instantiation & Forward Pass ===")

    # Instantiate the model
    model = model_lib.DPDB_HSE_CNN().to(device)
    print("Model instantiated: DPDB_HSE_CNN")

    # Get a batch from the loader
    images, angles, _, _ = next(iter(loader))
    images = images.to(device)
    angles = angles.to(device)

    # Forward pass
    outputs = model(images, angles)

    print(f"Model Output Shape: {outputs.shape}")

    # Verification
    assert outputs.shape == (images.size(0), 1), "Model output shape mismatch"

    # Verify DropBlock probability update logic
    print("Verifying DropBlock update logic...")
    initial_prob = model.dropblock3.drop_prob
    model.update_dropblock_prob(0.5)  # Simulate 50% training progress
    updated_prob = model.dropblock3.drop_prob

    print(f"DropBlock Prob: {initial_prob} -> {updated_prob}")
    assert updated_prob > initial_prob, "DropBlock probability did not increase"

    return model


def demo_training_loop(device, model, loader):
    print("\n=== Demo: Training Loop (One Epoch) ===")

    # Setup Optimizer and Loss
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    # Run one epoch using the library function
    # We pass total_epochs=2 to simulate being in the middle of training
    loss, acc = train_lib.train_one_epoch(
        model, loader, criterion, optimizer, device, epoch=0, total_epochs=2
    )

    print(f"Epoch 0 Result - Loss: {loss:.4f}, Accuracy: {acc:.4f}")

    # Verification
    assert not np.isnan(loss), "Training loss is NaN"
    assert 0.0 <= acc <= 1.0, "Accuracy is out of bounds"


def demo_inference_and_submission(device, model, loader):
    print("\n=== Demo: Inference and Submission Generation ===")

    # Run inference using the library function
    # predict_test expects a loader. We reuse the mini-loader.
    # Note: predict_test ignores labels in the loader.
    preds = train_lib.predict_test(model, loader, device)

    print(f"Predictions generated. Shape: {preds.shape}")
    print(f"Sample predictions: {preds[:5]}")

    # Verification
    assert preds.ndim == 1, "Predictions should be a 1D array"
    assert len(preds) == len(loader.dataset), "Prediction count mismatch"
    assert np.all(
        (preds >= 0) & (preds <= 1)
    ), "Predictions out of probability range [0, 1]"

    # Generate Submission File
    output_path = "./working/demo_submission.csv"
    ids = loader.dataset.ids

    utils.generate_submission_file(preds, ids, output_path)

    # Verify file creation
    assert os.path.exists(output_path), "Submission file was not created"

    # Verify file content
    df = pd.read_csv(output_path)
    print(f"Submission file loaded. Rows: {len(df)}")
    assert list(df.columns) == ["id", "is_iceberg"], "Incorrect submission columns"
    assert len(df) == len(ids), "Submission row count mismatch"

    print("Submission file verified successfully.")


if __name__ == "__main__":
    # 1. Setup
    utils.set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running demo on device: {device}")

    # 2. Data Loading
    loader = demo_data_loading()

    # 3. Model Logic
    model = demo_model_logic(device, loader)

    # 4. Training
    demo_training_loop(device, model, loader)

    # 5. Inference & Submission
    demo_inference_and_submission(device, model, loader)

    print("\nAll demonstrations completed successfully.")
