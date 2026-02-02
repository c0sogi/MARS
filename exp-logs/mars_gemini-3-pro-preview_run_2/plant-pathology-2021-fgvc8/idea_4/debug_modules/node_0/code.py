import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, calculate_metric
from library.losses import AsymmetricLoss
from library.dataset import AppleDataset, get_transforms, load_metadata
from library.models import AppleClassifier
from library.engine import train_one_epoch, validate, generate_submission


def run_demo():
    print("=== Apple Disease Detection Pipeline Demo ===\n")

    # 1. Configuration Setup
    print("[1] Configuring environment...")
    seed_everything(Config.seed)

    # Override Config for a fast demonstration run
    Config.debug = True  # Use only 100 samples per dataset
    Config.epochs = 1  # Train for only 1 epoch
    Config.batch_size = 4  # Small batch size
    Config.gradient_accumulation_steps = 1
    Config.num_workers = 2

    # Use a lightweight model for the demo to ensure speed
    demo_model_name = "resnet18"
    device = Config.device
    print(f"    Device: {device}")
    print(f"    Debug Mode: {Config.debug}")
    print(f"    Model: {demo_model_name}")

    # 2. Data Loading & Verification
    print("\n[2] Loading Data and Verifying Loaders...")

    # Load metadata
    train_df = load_metadata("train")
    val_df = load_metadata("val")
    test_df = load_metadata("test")

    # Instantiate Datasets (debug=True forces sampling of 100 images)
    train_dataset = AppleDataset(
        train_df, transforms=get_transforms("train"), debug=True
    )
    val_dataset = AppleDataset(val_df, transforms=get_transforms("valid"), debug=True)
    test_dataset = AppleDataset(test_df, transforms=get_transforms("test"), debug=True)

    print(f"    Train Dataset Size: {len(train_dataset)}")
    print(f"    Val Dataset Size:   {len(val_dataset)}")
    print(f"    Test Dataset Size:  {len(test_dataset)}")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
    )

    # Fetch one batch to verify shapes
    images, targets = next(iter(train_loader))
    print(f"    Batch Images Shape: {images.shape}")
    print(f"    Batch Targets Shape: {targets.shape}")

    # Assertions for data integrity
    assert images.shape == (
        Config.batch_size,
        3,
        Config.img_size,
        Config.img_size,
    ), "Image batch shape mismatch"
    assert targets.shape == (
        Config.batch_size,
        Config.num_classes,
    ), "Target batch shape mismatch"
    assert images.dtype == torch.float32
    assert targets.dtype == torch.float32

    # 3. Model & Loss Verification
    print("\n[3] Initializing Model and Loss...")

    model = AppleClassifier(model_name=demo_model_name, pretrained=True)
    model.to(device)

    criterion = AsymmetricLoss()

    # Dummy forward pass
    model.eval()
    with torch.no_grad():
        dummy_input = images.to(device)
        dummy_targets = targets.to(device)
        outputs = model(dummy_input)
        loss = criterion(outputs, dummy_targets)

    print(f"    Model Output Shape: {outputs.shape}")
    print(f"    Calculated Loss: {loss.item():.4f}")

    assert outputs.shape == (
        Config.batch_size,
        Config.num_classes,
    ), "Model output dimension incorrect"
    assert loss.ndim == 0, "Loss must be a scalar"
    assert not torch.isnan(loss), "Loss returned NaN"

    # 4. Training Loop Demo
    print("\n[4] Running Training Loop (1 Epoch)...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Train one epoch
    train_loss = train_one_epoch(
        model=model,
        optimizer=optimizer,
        data_loader=train_loader,
        device=device,
        criterion=criterion,
        epoch=1,
    )

    # Validate
    val_loss, val_f1 = validate(
        model=model, data_loader=val_loader, device=device, criterion=criterion
    )

    print(
        f"    Epoch 1 Summary -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}"
    )

    # 5. Metric Calculation Verification
    print("\n[5] Verifying Metric Calculation Logic...")
    # Create dummy perfect predictions
    y_true_dummy = np.array([[1, 0, 1], [0, 1, 0]])
    # Logits that will result in the same binary output after sigmoid (>0.5)
    y_pred_dummy = np.array([[0.9, 0.1, 0.8], [0.2, 0.8, 0.1]])

    metric_score = calculate_metric(y_true_dummy, y_pred_dummy, threshold=0.5)
    print(f"    Dummy F1 Score (Expected 1.0): {metric_score}")
    assert metric_score == 1.0, "Metric calculation failed for perfect match"

    # 6. Submission Generation
    print("\n[6] Generating Submission...")

    submission_path = "./working/demo_submission.csv"

    # Note: We must pass the DataFrame from the dataset object because
    # debug=True subsets the data. Passing the full test_df would cause a length mismatch.
    generate_submission(
        model=model,
        data_loader=test_loader,
        test_df=test_dataset.df,
        device=device,
        output_path=submission_path,
    )

    # Verify file existence and content
    assert os.path.exists(submission_path), "Submission file was not created"

    sub_df = pd.read_csv(submission_path)
    print(f"    Submission saved to {submission_path}")
    print(f"    Rows: {len(sub_df)}")
    print(f"    Columns: {list(sub_df.columns)}")
    print(f"    Sample:\n{sub_df.head(2)}")

    assert (
        "image" in sub_df.columns and "labels" in sub_df.columns
    ), "Submission columns missing"
    assert len(sub_df) == len(test_dataset), "Submission row count mismatch"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
