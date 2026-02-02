import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Import components from the provided library
from library.config import SEED, SUBMISSION_PATH
from library.utils import seed_everything, get_device, calculate_score, save_submission
from library.data_loader import get_dataloaders
from library.model import WBPA_Net
from library.train import fit_model


def run_demo():
    print("=== Iceberg Classification Pipeline Demo ===")

    # 1. Setup Environment
    # ----------------------------------------------------------------
    print("\n[1] Setting up environment...")
    seed_everything(SEED)
    device = get_device()
    print(f"    Device: {device}")

    # 2. Verify Utility Functions
    # ----------------------------------------------------------------
    print("\n[2] Verifying Utils...")
    # Test metric calculation
    y_true_dummy = [0, 1, 1, 0]
    y_pred_dummy = [0.1, 0.9, 0.8, 0.2]
    score = calculate_score(y_true_dummy, y_pred_dummy)
    print(f"    Dummy LogLoss: {score:.4f}")

    # Assert reasonable score logic
    assert (
        score < 1.0
    ), "LogLoss calculation yielded unexpectedly high value for good predictions."
    assert score >= 0.0, "LogLoss cannot be negative."

    # 3. Data Loading
    # ----------------------------------------------------------------
    print("\n[3] Loading Data...")
    # Using a small batch size for demonstration purposes
    BATCH_SIZE_DEMO = 8

    # get_dataloaders handles caching, splitting, and loader creation
    train_loader, val_loader, test_loader, ids_test = get_dataloaders(
        batch_size=BATCH_SIZE_DEMO,
        num_workers=0,  # Use 0 workers for simple debugging/demo to avoid multiprocessing overhead
        load_cached_data=True,
    )

    # Inspect the first batch from the training loader
    # The dataset returns ((image, angle), label)
    inputs_batch, targets_batch = next(iter(train_loader))
    images_batch, angles_batch = inputs_batch

    print(f"    Train Batch Images Shape: {images_batch.shape}")
    print(f"    Train Batch Angles Shape: {angles_batch.shape}")
    print(f"    Train Batch Targets Shape: {targets_batch.shape}")

    # Assertions to ensure data pipeline is correct
    # Expected: (Batch, 3, 75, 75) for images (Band1, Band2, Avg)
    assert images_batch.shape == (
        BATCH_SIZE_DEMO,
        3,
        75,
        75,
    ), "Image tensor shape mismatch."
    assert angles_batch.shape == (BATCH_SIZE_DEMO,), "Angle tensor shape mismatch."
    assert targets_batch.shape == (BATCH_SIZE_DEMO,), "Target tensor shape mismatch."

    # 4. Model Initialization
    # ----------------------------------------------------------------
    print("\n[4] Initializing WBPA_Net Model...")
    model = WBPA_Net().to(device)

    # Perform a dummy forward pass to verify architecture
    images_gpu = images_batch.to(device)
    angles_gpu = angles_batch.to(device)

    with torch.no_grad():
        outputs_dummy = model(images_gpu, angles_gpu)

    print(f"    Model Output Shape: {outputs_dummy.shape}")

    # Expecting (Batch, 1) raw logits
    assert outputs_dummy.shape == (BATCH_SIZE_DEMO, 1), "Model output shape mismatch."

    # 5. Training Loop Execution
    # ----------------------------------------------------------------
    print("\n[5] Running Training Loop (Fast Demo)...")

    # Define Optimizer and Loss
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    # Run fit_model for a limited number of epochs to demonstrate functionality quickly
    # fit_model handles the training loop, validation, and early stopping
    trained_model = fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=2,  # Limited for speed
        patience=1,
    )

    print("    Training complete.")

    # 6. Inference and Submission
    # ----------------------------------------------------------------
    print("\n[6] Generating Submission...")
    trained_model.eval()
    all_preds = []

    # Iterate over test loader
    # Test loader returns (images, angles) without labels
    with torch.no_grad():
        for inputs in test_loader:
            images_test, angles_test = inputs

            images_test = images_test.to(device)
            angles_test = angles_test.to(device)

            # Forward pass
            logits = trained_model(images_test, angles_test)

            # Apply Sigmoid to get probabilities
            probs = torch.sigmoid(logits)

            all_preds.extend(probs.cpu().numpy().flatten())

    # Verify predictions match the number of test IDs
    print(f"    Generated {len(all_preds)} predictions.")
    assert len(all_preds) == len(
        ids_test
    ), f"Mismatch: {len(all_preds)} preds vs {len(ids_test)} IDs."

    # Save submission using utility function
    save_submission(ids_test, all_preds, SUBMISSION_PATH)

    # Verify file creation
    assert os.path.exists(SUBMISSION_PATH), "Submission file was not created."
    print(f"    Submission saved successfully to: {SUBMISSION_PATH}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
