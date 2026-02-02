import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

# Import functions and classes from the provided library files
from library.utils import seed_everything, get_device
from library.data import get_dataloaders
from library.model import IcebergResNet
from library.train import train_one_epoch, validate_one_epoch
from library.predict import predict_with_tta, generate_ensemble_submission


def run_demonstration():
    print("=== Starting Iceberg Classification Workflow Demo ===")

    # 1. Setup Environment
    # Ensure reproducibility
    seed_everything(42)
    device = get_device()
    print(f"Compute Device: {device}")

    # Define working directory for model artifacts (matches library default)
    working_dir = "./working/idea_5/"
    os.makedirs(working_dir, exist_ok=True)

    # 2. Data Loading
    print("\n[Step 1] Initializing DataLoaders...")
    # We use a small batch size. load_cached_data=False forces the processing logic to run,
    # demonstrating how raw JSONs are converted to tensors.
    batch_size = 16
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size, num_workers=2, load_cached_data=False
    )

    # Verify Data Integrity
    print("Verifying Train Loader batch...")
    images, angles, labels = next(iter(train_loader))

    # Check shapes: Images (B, 3, 224, 224), Angles (B,), Labels (B,)
    assert images.shape == (
        batch_size,
        3,
        224,
        224,
    ), f"Unexpected image shape: {images.shape}"
    assert angles.shape == (batch_size,), f"Unexpected angle shape: {angles.shape}"
    assert labels.shape == (batch_size,), f"Unexpected label shape: {labels.shape}"

    # Check value ranges
    assert (
        images.min() >= 0.0 and images.max() <= 1.0
    ), "Images should be normalized to [0, 1]"

    print("Verifying Test Loader batch...")
    t_images, t_angles, t_ids = next(iter(test_loader))
    assert t_images.shape == (batch_size, 3, 224, 224)
    assert len(t_ids) == batch_size
    print("Data loading verified successfully.")

    # 3. Model Initialization
    print("\n[Step 2] Initializing Model...")
    model = IcebergResNet().to(device)

    # Verify Forward Pass
    images = images.to(device)
    angles = angles.to(device)

    logits = model(images, angles)
    assert logits.shape == (batch_size, 1), f"Output shape mismatch: {logits.shape}"
    print("Model forward pass successful.")

    # 4. Training Loop Demonstration
    print("\n[Step 3] Running Training Loop (1 Epoch)...")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # Train for one epoch
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    print(f"Epoch 1 Train Loss: {train_loss:.4f}")
    assert train_loss > 0, "Training loss should be positive."

    # Validate
    val_loss, val_log_loss, val_acc = validate_one_epoch(
        model, val_loader, criterion, device
    )
    print(
        f"Epoch 1 Val Loss: {val_loss:.4f} | LogLoss: {val_log_loss:.4f} | Acc: {val_acc:.4f}"
    )
    assert val_log_loss > 0, "Log loss should be positive."

    # 5. Inference and Submission
    print("\n[Step 4] Generating Predictions and Submission...")

    # Test Time Augmentation (TTA) Prediction
    ids, preds = predict_with_tta(model, test_loader, device)

    # Verify predictions
    assert len(ids) == len(test_loader.dataset)
    assert len(preds) == len(test_loader.dataset)
    assert (preds >= 0).all() and (
        preds <= 1
    ).all(), "Predictions must be probabilities [0, 1]"
    print(f"Generated {len(preds)} predictions.")

    # Demonstrate Ensemble Submission Generation
    # First, save the current model as if it were 'Fold 0'
    model_path = os.path.join(working_dir, "model_fold_0_best.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Saved checkpoint to {model_path}")

    # Use the library function to generate submission from saved models
    # We set n_folds=1 to use just the model we saved
    generate_ensemble_submission(
        n_folds=1, batch_size=batch_size, model_dir=working_dir, load_cached_data=True
    )

    # Verify Submission File
    submission_path = "./submission/submission.csv"
    assert os.path.exists(submission_path), "Submission file was not created."

    df_sub = pd.read_csv(submission_path)
    print(f"Submission file loaded. Shape: {df_sub.shape}")
    print(df_sub.head())

    assert list(df_sub.columns) == ["id", "is_iceberg"], "Incorrect submission columns."
    assert len(df_sub) == 321, "Incorrect number of rows in submission."

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demonstration()
