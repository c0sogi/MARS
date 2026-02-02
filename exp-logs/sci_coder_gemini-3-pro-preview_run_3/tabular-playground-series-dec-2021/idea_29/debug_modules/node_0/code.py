import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

# Import provided library components
from library.utils import seed_everything, get_device, generate_submission
from library.data_loader import get_dataloaders
from library.model import DeepParallelVectorDCNResNet
from library.train import train_one_epoch, validate


def main():
    print("Starting Demo Pipeline...")

    # 1. Setup Environment
    # --------------------------------------------------------------------------
    seed_everything(42)
    device = get_device()
    print(f"Device: {device}")

    # Define working directories for the demo
    base_dir = "./working/demo_pipeline"
    cache_dir = os.path.join(base_dir, "cache")
    submission_dir = os.path.join(base_dir, "submission")
    submission_path = os.path.join(submission_dir, "submission.csv")

    # Clean up previous runs to ensure fresh execution
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(submission_dir, exist_ok=True)

    # 2. Data Pipeline Verification
    # --------------------------------------------------------------------------
    print("\n[1/4] Verifying Data Loading and Feature Engineering...")

    # Load data using the library function.
    # We force processing (load_cached_data=False) to test the feature engineering logic.
    # Using a larger batch size here for the loader creation, but we will subset later.
    train_loader, val_loader, test_loader, test_ids, input_dim = get_dataloaders(
        load_cached_data=False,
        batch_size=2048,
        data_dir="./metadata",
        cache_dir=cache_dir,
    )

    print(f"  Input Feature Dimension: {input_dim}")

    # Validation: Check input dimensions
    # Original cols (54) + engineered features should result in > 54
    assert input_dim >= 54, "Feature engineering failed to produce expected dimensions."

    # Validation: Check batch structure
    sample_inputs, sample_targets = next(iter(train_loader))
    print(
        f"  Sample Batch Shape - Inputs: {sample_inputs.shape}, Targets: {sample_targets.shape}"
    )

    assert sample_inputs.shape[1] == input_dim, "Batch input dimension mismatch."
    assert sample_targets.max() < 7, "Targets should be in range 0-6 (mapped from 1-7)."

    # Create Mini-Datasets for speed
    # We don't want to iterate over 2.8M rows for a demo.
    print("  Creating mini-datasets for rapid testing...")
    mini_train_ds = Subset(train_loader.dataset, range(1000))
    mini_val_ds = Subset(val_loader.dataset, range(500))
    mini_test_ds = Subset(test_loader.dataset, range(100))

    mini_train_loader = DataLoader(mini_train_ds, batch_size=100, shuffle=True)
    mini_val_loader = DataLoader(mini_val_ds, batch_size=100, shuffle=False)
    mini_test_loader = DataLoader(mini_test_ds, batch_size=100, shuffle=False)
    mini_test_ids = test_ids[:100]

    # 3. Model Architecture Verification
    # --------------------------------------------------------------------------
    print("\n[2/4] Verifying Model Architecture...")

    # Instantiate model with reduced capacity for speed
    model = DeepParallelVectorDCNResNet(
        input_dim=input_dim,
        num_classes=7,
        hidden_dim=64,  # Reduced from 512
        num_cross_layers=1,  # Reduced from 3
        num_res_blocks=1,  # Reduced from 4
        dropout_rate=0.1,
    ).to(device)

    # Validation: Forward pass with dummy data
    dummy_input = torch.randn(5, input_dim).to(device)
    dummy_output = model(dummy_input)

    print(f"  Model Output Shape: {dummy_output.shape}")
    assert dummy_output.shape == (5, 7), "Model output shape mismatch."

    # 4. Training Loop Verification
    # --------------------------------------------------------------------------
    print("\n[3/4] Verifying Training and Validation Logic...")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Run one training epoch on mini-set
    train_loss, train_acc = train_one_epoch(
        model, mini_train_loader, criterion, optimizer, device
    )
    print(f"  Train Step -> Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

    assert not np.isnan(train_loss), "Training loss is NaN."
    assert 0 <= train_acc <= 1, "Training accuracy out of bounds."

    # Run validation step on mini-set
    val_loss, val_acc = validate(model, mini_val_loader, criterion, device)
    print(f"  Val Step   -> Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

    assert not np.isnan(val_loss), "Validation loss is NaN."

    # 5. Inference Verification
    # --------------------------------------------------------------------------
    print("\n[4/4] Verifying Submission Generation...")

    generate_submission(
        model, mini_test_loader, mini_test_ids, device, output_path=submission_path
    )

    # Validation: Check submission file
    assert os.path.exists(submission_path), "Submission file was not created."

    df_sub = pd.read_csv(submission_path)
    print(f"  Submission File Shape: {df_sub.shape}")
    print(f"  First 3 rows:\n{df_sub.head(3)}")

    assert df_sub.shape == (100, 2), f"Expected (100, 2), got {df_sub.shape}"
    assert list(df_sub.columns) == [
        "Id",
        "Cover_Type",
    ], "Incorrect columns in submission."
    assert df_sub["Cover_Type"].dtype == np.int64, "Cover_Type should be integers."

    print("\nDemo Pipeline Completed Successfully.")


if __name__ == "__main__":
    main()
