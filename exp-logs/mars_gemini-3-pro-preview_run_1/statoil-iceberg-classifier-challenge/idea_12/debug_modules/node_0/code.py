import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import warnings

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, calculate_log_loss
from library.data import get_dataloaders
from library.model import IcebergResNet18
from library.engine import train_one_epoch, evaluate, predict
from library.calibration import PlattScaler


def run_demo():
    # 1. Setup and Configuration Overrides for Demo Speed
    print("--- Setting up Demo Configuration ---")

    # Override Config for speed
    Config.MAX_EPOCHS = 2
    Config.BATCH_SIZE = 32
    Config.NUM_WORKERS = 2  # Reduce workers to avoid overhead in short run
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.CALIBRATION_PATH = os.path.join(Config.WORKING_DIR, "calibration_model.pkl")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    # Set seed for reproducibility
    seed_everything(Config.SEED)

    device = Config.DEVICE
    print(f"Device: {device}")

    # 2. Data Loading
    print("\n--- Loading Data ---")
    # We use load_cached_data=False to force processing raw json for the demo
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=False, full_fit=False
    )

    # Verify Data Shapes
    images, angles, labels = next(iter(train_loader))
    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Angle Shape: {angles.shape}")
    print(f"Batch Label Shape: {labels.shape}")

    # Assertions for data integrity
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), "Incorrect image tensor shape"
    assert angles.shape[0] == Config.BATCH_SIZE, "Incorrect angle tensor shape"
    assert labels.shape[0] == Config.BATCH_SIZE, "Incorrect label tensor shape"

    # 3. Model Initialization
    print("\n--- Initializing Model ---")
    model = IcebergResNet18()
    model.to(device)

    # Verify Forward Pass
    dummy_images = images.to(device)
    dummy_angles = angles.to(device)
    with torch.no_grad():
        dummy_logits = model(dummy_images, dummy_angles)

    print(f"Logits Shape: {dummy_logits.shape}")
    assert dummy_logits.shape == (Config.BATCH_SIZE, 1), "Output logits must be (B, 1)"

    # 4. Training Loop (Short Demo)
    print("\n--- Starting Training (Demo: 2 Epochs) ---")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    best_loss = float("inf")

    for epoch in range(1, Config.MAX_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)

        # 5. Evaluation
        val_loss, val_metric, val_logits, val_targets = evaluate(
            model, val_loader, device
        )

        print(
            f"Epoch {epoch} Summary: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val LogLoss={val_metric:.4f}"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            # Save checkpoint (mock save)
            torch.save(
                model.state_dict(),
                os.path.join(Config.CHECKPOINT_DIR, "best_model.pth"),
            )

    # 6. Calibration (Platt Scaling)
    print("\n--- Calibrating Model ---")
    # We use the logits from the last validation epoch for demonstration
    # In a real scenario, this should be OOF logits from cross-validation
    scaler = PlattScaler()
    scaler.fit(val_logits, val_targets)

    # Verify scaler
    calibrated_probs = scaler.predict(val_logits)
    calibrated_loss = calculate_log_loss(val_targets, calibrated_probs)
    print(f"Calibrated Validation Log Loss: {calibrated_loss:.4f}")

    scaler.save()

    # Reload scaler to verify persistence
    scaler_loaded = PlattScaler().load()
    assert scaler_loaded.is_fitted, "Loaded scaler should be fitted"

    # 7. Inference on Test Set
    print("\n--- Generating Predictions ---")
    # Load best model weights
    model.load_state_dict(
        torch.load(
            os.path.join(Config.CHECKPOINT_DIR, "best_model.pth"), weights_only=True
        )
    )

    test_ids, test_probs_raw = predict(model, test_loader, device)

    # Convert raw probabilities (sigmoid output) back to logits for the scaler
    # Or use predict_from_probs if the model output was already sigmoid-ed in predict()
    # Note: library.engine.predict returns probabilities (torch.sigmoid applied).
    # PlattScaler.predict_from_probs handles the conversion back to logits internally.
    test_probs_calibrated = scaler.predict_from_probs(test_probs_raw)

    print(f"Test Predictions Shape: {test_probs_calibrated.shape}")
    assert len(test_ids) == len(
        test_probs_calibrated
    ), "Mismatch between IDs and predictions"

    # 8. Create Submission
    print("\n--- Creating Submission File ---")
    submission_df = pd.DataFrame({"id": test_ids, "is_iceberg": test_probs_calibrated})

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print("Head of submission:")
    print(submission_df.head())

    # Final assertion to check file existence
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"
    print("\nDemo completed successfully.")


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")
    run_demo()
