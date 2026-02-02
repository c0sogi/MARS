import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.optim.swa_utils import AveragedModel

# Import library components
from library.config import Config
from library.utils import set_seed, get_scheduler, save_checkpoint
from library.data import get_loaders
from library.model import IcebergResNet18
from library.engine import (
    train_one_epoch,
    evaluate,
    run_swa_step,
    update_bn,
    predict_tta,
)
from library.calibration import PlattScaler


def main():
    # ==========================================
    # 1. Setup & Configuration Override
    # ==========================================
    print("Initializing Demo...")

    # Set seed for reproducibility
    set_seed(42)

    # Override Config for rapid demonstration
    Config.DEBUG = True  # Use small data subset
    Config.BATCH_SIZE = 8  # Small batch size for demo
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.MAX_EPOCHS_PHASE_1 = 2  # Minimal epochs
    Config.SWA_DURATION = 1  # Minimal SWA duration

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ==========================================
    # 2. Data Loading
    # ==========================================
    print("\nLoading Data...")
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=False)

    # Verification: Check batch structure
    sample_batch = next(iter(train_loader))
    images = sample_batch["image"]
    angles = sample_batch["angle"]
    labels = sample_batch["label"]

    # Expected shapes: (B, 3, 224, 224), (B,), (B,)
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        224,
        224,
    ), f"Image shape mismatch: {images.shape}"
    assert angles.shape == (Config.BATCH_SIZE,), f"Angle shape mismatch: {angles.shape}"
    assert labels.shape == (Config.BATCH_SIZE,), f"Label shape mismatch: {labels.shape}"
    print("Data loaded and verified successfully.")

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    print("\nInitializing Model...")
    model = IcebergResNet18().to(device)

    # Verification: Forward pass
    with torch.no_grad():
        dummy_out = model(images.to(device), angles.to(device))
    assert dummy_out.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Output shape mismatch: {dummy_out.shape}"
    print("Model initialized and forward pass verified.")

    # ==========================================
    # 4. Training Loop (Phase 1)
    # ==========================================
    print("\nStarting Training (Phase 1)...")
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = get_scheduler(optimizer, mode="plateau")

    best_loss = float("inf")

    for epoch in range(1, Config.MAX_EPOCHS_PHASE_1 + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)

        # Evaluate
        val_loss, val_logits, val_targets = evaluate(model, val_loader, device)

        # Scheduler Step
        scheduler.step(val_loss)

        # Checkpoint logic
        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                is_best=True,
            )

        print(f"Epoch {epoch} finished. Val Loss: {val_loss:.4f}")

    # ==========================================
    # 5. SWA (Stochastic Weight Averaging) Demo
    # ==========================================
    print("\nRunning SWA Step...")
    # Initialize SWA model
    swa_model = AveragedModel(model).to(device)

    # Simulate one SWA update step using the engine function
    run_swa_step(swa_model, model)

    # Update BatchNorm statistics
    print("Updating SWA BatchNorm statistics...")
    update_bn(train_loader, swa_model, device)
    print("SWA step completed.")

    # ==========================================
    # 6. Calibration (Platt Scaling)
    # ==========================================
    print("\nCalibrating Model...")
    # Get validation logits from the best model (or current model for demo)
    _, val_logits, val_targets = evaluate(model, val_loader, device)

    scaler = PlattScaler()
    scaler.fit(val_logits, val_targets)

    # Verify calibration output
    probs = scaler.predict_proba(val_logits)
    assert (
        probs.min() >= 0.0 and probs.max() <= 1.0
    ), "Probabilities out of bounds [0, 1]"
    assert probs.shape == val_targets.shape, "Probability shape mismatch"

    # Test Save/Load
    scaler.save("demo_scaler.npz")
    scaler_new = PlattScaler()
    scaler_new.load("demo_scaler.npz")
    assert np.isclose(scaler.A, scaler_new.A), "Scaler load failed (A mismatch)"
    print("Calibration verified.")

    # ==========================================
    # 7. Inference (TTA) & Submission
    # ==========================================
    print("\nRunning Inference (TTA)...")
    # Use the SWA model for inference (usually better)
    test_logits, test_ids = predict_tta(swa_model, test_loader, device)

    # Apply calibration
    test_probs = scaler.predict_proba(test_logits)

    # Create Submission DataFrame
    sub_df = pd.DataFrame({"id": test_ids, "is_iceberg": test_probs})

    # Verify submission format
    assert len(sub_df) == len(test_ids), "Submission length mismatch"
    assert (
        "id" in sub_df.columns and "is_iceberg" in sub_df.columns
    ), "Submission columns mismatch"

    # Save
    sub_path = Config.SUBMISSION_PATH
    sub_df.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")

    # Final check
    if os.path.exists(sub_path):
        print("Demo completed successfully.")
    else:
        raise FileNotFoundError("Submission file was not created.")


if __name__ == "__main__":
    main()
