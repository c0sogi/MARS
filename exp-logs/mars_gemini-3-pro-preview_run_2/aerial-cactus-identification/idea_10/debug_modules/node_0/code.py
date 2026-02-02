import os
import sys
import torch
import numpy as np
import pandas as pd

# Import from the provided library files
from library.config import set_seed, OUTPUT_DIR, MODEL_DIR, IDEA_DIR
from library.dataset import get_dataloaders
from library.model import CustomNarrowSEMultiScaleResNet, predict_with_tta
from library.train import train_single_seed
from library.inference import run_inference


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Setup & Reproducibility
    DEMO_SEED = 999
    set_seed(DEMO_SEED)
    print(f"Random seed set to {DEMO_SEED}.")

    # 2. Verify Data Loading
    print("\n--- Verifying Data Loading ---")
    # Use a small batch size for demonstration speed
    demo_batch_size = 16

    # get_dataloaders returns: train_loader, val_loader, test_loader, test_ids
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        batch_size=demo_batch_size,
        num_workers=0,  # Use 0 workers for simple debugging/demo to avoid multiprocessing overhead
        load_cached_data=True,
    )

    # Fetch a single batch from the training loader
    images, labels = next(iter(train_loader))

    # Assertions for shapes
    # Images: (Batch, Channels, Height, Width) -> (16, 3, 32, 32)
    assert images.dim() == 4, f"Expected 4D image tensor, got {images.dim()}"
    assert images.shape == (
        demo_batch_size,
        3,
        32,
        32,
    ), f"Expected shape {(demo_batch_size, 3, 32, 32)}, got {images.shape}"

    # Labels: (Batch,) -> (16,)
    assert labels.dim() == 1, f"Expected 1D label tensor, got {labels.dim()}"
    assert (
        labels.shape[0] == demo_batch_size
    ), f"Expected {demo_batch_size} labels, got {labels.shape[0]}"

    print("Data Loader verification passed: Batch shapes are correct.")

    # 3. Verify Model Architecture
    print("\n--- Verifying Model Architecture ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomNarrowSEMultiScaleResNet().to(device)

    # Move demo batch to device
    images = images.to(device)

    # Perform forward pass
    with torch.no_grad():
        outputs = model(images)

    # Assert output shape: (Batch, 1) because it's binary classification with Logits
    assert outputs.shape == (
        demo_batch_size,
        1,
    ), f"Expected output shape {(demo_batch_size, 1)}, got {outputs.shape}"

    # Verify TTA function
    tta_pred = predict_with_tta(model, images, device)
    assert tta_pred.shape == (demo_batch_size, 1), "TTA output shape mismatch"
    assert torch.all(tta_pred >= 0) and torch.all(
        tta_pred <= 1
    ), "TTA predictions should be probabilities (0-1)"

    print("Model verification passed: Forward pass and TTA successful.")

    # 4. Demonstrate Training
    print("\n--- Demonstrating Training Loop ---")
    # Train for just 1 epoch to verify the pipeline runs
    # We override batch_size to match our demo setting
    best_auc = train_single_seed(seed=DEMO_SEED, epochs=1, batch_size=32)

    # Check if model file was created
    expected_model_path = os.path.join(MODEL_DIR, f"model_seed_{DEMO_SEED}.pth")
    assert os.path.exists(
        expected_model_path
    ), f"Model checkpoint not found at {expected_model_path}"
    assert isinstance(best_auc, float), "Train function should return a float AUC score"

    print(f"Training demonstration passed. Model saved to {expected_model_path}")

    # 5. Demonstrate Inference
    print("\n--- Demonstrating Inference Pipeline ---")
    # Run inference using the model we just trained
    run_inference(seeds=[DEMO_SEED], batch_size=32, load_cached_data=True)

    # Check submission file
    submission_path = os.path.join(OUTPUT_DIR, "submission.csv")
    assert os.path.exists(
        submission_path
    ), f"Submission file not found at {submission_path}"

    # Verify submission content
    df_sub = pd.read_csv(submission_path)

    # Check columns
    assert list(df_sub.columns) == [
        "id",
        "has_cactus",
    ], f"Invalid columns: {df_sub.columns}"

    # Check length (should match test_ids count)
    assert len(df_sub) == len(
        test_ids
    ), f"Submission length mismatch. Expected {len(test_ids)}, got {len(df_sub)}"

    # Check value range
    assert df_sub["has_cactus"].min() >= 0.0, "Probabilities cannot be negative"
    assert df_sub["has_cactus"].max() <= 1.0, "Probabilities cannot exceed 1.0"

    print(f"Inference demonstration passed. Submission verified at {submission_path}")
    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
