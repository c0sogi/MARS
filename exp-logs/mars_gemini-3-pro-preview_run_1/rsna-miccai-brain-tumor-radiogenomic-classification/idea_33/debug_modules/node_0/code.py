import os
import sys
import torch
import pandas as pd
import numpy as np

# Import components from the provided library files
from library.config import WORKING_DIR, DEVICE, SEED
from library.utils import seed_everything, load_checkpoint
from library.data_processing import get_dataloaders
from library.model import RARVEfficientNet
from library.trainer import run_training


def main():
    # 1. Setup and Configuration
    # Ensure reproducibility
    seed_everything(SEED)

    print(f"Running on Device: {DEVICE}")
    print(f"Working Directory: {WORKING_DIR}")

    # 2. Data Pipeline Verification
    print("\n" + "=" * 40)
    print(" STEP 1: Verifying Data Pipeline")
    print("=" * 40)

    # Initialize DataLoaders
    # We use a small batch size for inspection and force cache generation
    # to ensure the ROI slicing logic executes correctly.
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=4, load_cached_data=False
    )

    # Fetch a single batch from the training loader
    try:
        images, targets, subject_ids = next(iter(train_loader))
    except StopIteration:
        raise RuntimeError("Train loader is empty!")

    # Verify Shapes and Types
    # Input should be (Batch, 9, 224, 224) corresponding to 3 depths x 3 modalities
    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Target Shape: {targets.shape}")

    assert images.shape == (
        4,
        9,
        224,
        224,
    ), f"Expected image shape (4, 9, 224, 224), got {images.shape}"
    assert targets.shape == (4,), f"Expected target shape (4,), got {targets.shape}"
    assert images.dtype == torch.float32, f"Expected float32 images, got {images.dtype}"

    print("Data Pipeline Verified: Shapes and types are correct.")

    # 3. Model Architecture Verification
    print("\n" + "=" * 40)
    print(" STEP 2: Verifying Model Architecture")
    print("=" * 40)

    # Instantiate the model
    model = RARVEfficientNet().to(DEVICE)

    # Move verification batch to device
    images = images.to(DEVICE)

    # Perform a forward pass (inference mode)
    model.eval()
    with torch.no_grad():
        outputs = model(images)

    print(f"Model Output Shape: {outputs.shape}")

    # Verify Output Shape (Binary Classification Logits -> (B, 1))
    assert outputs.shape == (4, 1), f"Expected output shape (4, 1), got {outputs.shape}"

    # Check for NaNs
    assert not torch.isnan(outputs).any(), "Model output contains NaNs"

    print("Model Architecture Verified: Forward pass successful.")

    # 4. Training Loop Execution
    print("\n" + "=" * 40)
    print(" STEP 3: Executing Training Loop (Demo)")
    print("=" * 40)

    # Run training for 1 epoch to demonstrate the trainer functionality.
    # This uses the default batch size defined in library.config (32).
    # load_cached_data=True reuses the cache generated in Step 1.
    print("Starting training for 1 epoch...")
    run_training(load_cached_data=True, max_epochs=1, patience=1)

    # Verify that checkpoints were saved
    best_model_path = os.path.join(WORKING_DIR, "best_model.pth")
    epoch_checkpoint_path = os.path.join(WORKING_DIR, "checkpoint_epoch_1.pth")

    # Determine which file to load for inference
    if os.path.exists(best_model_path):
        inference_checkpoint = "best_model.pth"
        print(f"Found best model checkpoint at: {best_model_path}")
    elif os.path.exists(epoch_checkpoint_path):
        inference_checkpoint = "checkpoint_epoch_1.pth"
        print(f"Found epoch checkpoint at: {epoch_checkpoint_path}")
    else:
        raise FileNotFoundError("Training finished but no checkpoint file was found.")

    # 5. Inference and Submission
    print("\n" + "=" * 40)
    print(" STEP 4: Inference and Submission Generation")
    print("=" * 40)

    # Load the trained model
    inference_model = RARVEfficientNet().to(DEVICE)
    load_checkpoint(inference_model, filename=inference_checkpoint, device=DEVICE)
    inference_model.eval()

    predictions = []
    ids_list = []

    print("Running inference on Test Set...")
    with torch.no_grad():
        for images, _, ids in test_loader:
            images = images.to(DEVICE)

            # Forward pass
            logits = inference_model(images)

            # Apply Sigmoid to get probabilities
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            predictions.extend(probs)
            ids_list.extend(ids)

    # Create Submission DataFrame
    submission = pd.DataFrame({"BraTS21ID": ids_list, "MGMT_value": predictions})

    # Verify Submission Format
    print("First 5 predictions:")
    print(submission.head())

    assert (
        "BraTS21ID" in submission.columns and "MGMT_value" in submission.columns
    ), "Submission columns are missing."
    assert len(submission) == len(
        test_loader.dataset
    ), f"Submission length {len(submission)} does not match test set size {len(test_loader.dataset)}."
    assert (
        submission["MGMT_value"].min() >= 0.0 and submission["MGMT_value"].max() <= 1.0
    ), "Probabilities must be between 0 and 1."

    # Save Submission
    submission_path = os.path.join(WORKING_DIR, "submission.csv")
    submission.to_csv(submission_path, index=False)
    print(f"\nSubmission saved successfully to {submission_path}")

    print("\nAll demonstration steps completed successfully.")


if __name__ == "__main__":
    main()
