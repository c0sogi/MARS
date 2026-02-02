import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import from the provided library files
from library.utils import set_seed, get_device
from library.dataset import get_datasets
from library.model import SIA_DS_EfficientNet
from library.training import run_fold
from library.inference import predict_test_set


def run_demo():
    # 1. Setup and Configuration
    print(">>> Step 1: Setup and Configuration")
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    # Define paths
    metadata_dir = "./metadata"
    working_dir = "./working"
    submission_path = os.path.join(working_dir, "submission.csv")
    model_save_path = os.path.join(working_dir, "best_model_fold0.pth")

    os.makedirs(working_dir, exist_ok=True)

    # 2. Data Loading & Verification
    print("\n>>> Step 2: Data Loading & Verification")
    # Load a small subset of data (limit_size=10) for demonstration speed
    limit_size = 10
    train_ds, val_ds, test_ds = get_datasets(
        metadata_dir=metadata_dir,
        load_cached_data=False,  # Force processing to demo logic
        limit_size=limit_size,
    )

    print(f"Train set size: {len(train_ds)}")
    print(f"Val set size: {len(val_ds)}")
    print(f"Test set size: {len(test_ds)}")

    # Verify Dataset Output
    # The dataset should return a tensor of shape (9, 224, 224) and a label/ID
    sample_img, sample_label = train_ds[0]

    print(f"Sample Image Shape: {sample_img.shape}")
    print(f"Sample Label: {sample_label}")

    # Assertions
    assert len(train_ds) > 0, "Training dataset is empty."
    assert sample_img.shape == (
        9,
        224,
        224,
    ), f"Expected shape (9, 224, 224), got {sample_img.shape}"
    assert isinstance(sample_img, torch.Tensor), "Output image is not a Tensor."

    # 3. Model Initialization & Verification
    print("\n>>> Step 3: Model Initialization")
    model = SIA_DS_EfficientNet(num_classes=1, drop_rate=0.3)
    model = model.to(device)

    # Verify the first layer adaptation (9 input channels)
    first_layer = model.backbone.conv_stem
    print(f"First Layer In-Channels: {first_layer.in_channels}")
    print(f"First Layer Weight Shape: {first_layer.weight.shape}")

    # Assertions
    assert first_layer.in_channels == 9, "Model first layer does not accept 9 channels."
    assert (
        first_layer.weight.shape[1] == 9
    ), "Model weights not initialized for 9 channels."

    # Test Forward Pass
    dummy_input = sample_img.unsqueeze(0).to(device)  # Add batch dim
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Forward pass output shape: {output.shape}")
    assert output.shape == (1, 1), "Output shape mismatch for binary classification."

    # 4. Training Loop Demonstration
    print("\n>>> Step 4: Training Loop (Mini-Run)")
    batch_size = 4
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Run training for 1 epoch
    best_auc = run_fold(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=1,
        patience=1,
        save_path=model_save_path,
    )

    print(f"Training completed. Best AUC: {best_auc}")
    assert os.path.exists(model_save_path), "Model checkpoint was not saved."

    # 5. Inference Demonstration
    print("\n>>> Step 5: Inference Pipeline")

    # We use the predict_test_set function from library.inference
    # It expects models named best_model_foldX.pth in the model_dir
    # We saved our demo model as best_model_fold0.pth, so we set num_folds=1

    predict_test_set(
        model_dir=working_dir,
        output_path=submission_path,
        metadata_dir=metadata_dir,
        batch_size=batch_size,
        num_folds=1,
        load_cached_data=True,  # Use the cache generated in Step 2
        limit_size=limit_size,
    )

    # Verify Submission
    assert os.path.exists(submission_path), "Submission file was not generated."

    df_sub = pd.read_csv(submission_path)
    print("Submission File Head:")
    print(df_sub.head())

    assert "BraTS21ID" in df_sub.columns, "Submission missing BraTS21ID column."
    assert "MGMT_value" in df_sub.columns, "Submission missing MGMT_value column."
    assert len(df_sub) > 0, "Submission file is empty."

    print("\n>>> Demo Completed Successfully!")


if __name__ == "__main__":
    run_demo()
