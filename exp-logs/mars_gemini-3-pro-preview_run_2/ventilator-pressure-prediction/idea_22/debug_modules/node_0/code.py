import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import library components
from library.config import Config
from library.utils import seed_everything
from library.data_factory import get_dataloaders
from library.model_architecture import PADIBiLSTM, WeightedL1Loss
from library.training_engine import Trainer


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print(">>> [1/6] Setting up configuration for demo execution...")

    # Override Config for a fast, isolated debug run
    Config.EXP_NAME = "demo_execution_script"
    Config.WORKING_DIR = os.path.join("./working", Config.EXP_NAME)
    Config.DEBUG = True  # Use small data subset
    Config.BATCH_SIZE = 16  # Small batch size for debug
    Config.EPOCHS = 2  # Minimal epochs
    Config.NUM_WORKERS = 0  # simple execution

    # Create working directory
    Config.setup()

    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Device: {device}")
    print(f"    Working Directory: {Config.WORKING_DIR}")

    # ==========================================
    # 2. Data Loading & Verification
    # ==========================================
    print("\n>>> [2/6] Initializing DataLoaders and Verifying Data Shapes...")

    # Force load_cached_data=False to demonstrate the feature engineering pipeline
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=False, debug=Config.DEBUG
    )

    # Fetch one batch to verify structure
    batch = next(iter(train_loader))
    X, y, u_out = batch["X"], batch["y"], batch["u_out"]

    print(f"    Batch X (Features) shape: {X.shape}")
    print(f"    Batch y (Targets) shape:  {y.shape}")
    print(f"    Batch u_out (Mask) shape: {u_out.shape}")

    # Assertions to ensure data integrity
    assert X.ndim == 3, "X should be (Batch, Seq, Features)"
    assert y.ndim == 2, "y should be (Batch, Seq)"
    assert X.shape[1] == 80, "Sequence length should be 80"
    assert X.shape[2] == Config.INPUT_DIM, f"Feature dim should be {Config.INPUT_DIM}"

    # ==========================================
    # 3. Model Initialization & Forward Pass
    # ==========================================
    print("\n>>> [3/6] Instantiating Model and Running Forward Pass...")

    model = PADIBiLSTM().to(device)

    # Move sample batch to device
    X_dev = X.to(device)
    y_dev = y.to(device)
    u_out_dev = u_out.to(device)

    # Forward pass
    preds = model(X_dev)

    print(f"    Prediction shape: {preds.shape}")

    # Assert output shape matches target
    assert preds.shape == y_dev.shape, "Model output shape mismatch"

    # Verify Loss Calculation
    criterion = WeightedL1Loss()
    loss = criterion(preds, y_dev, u_out_dev)
    print(f"    Initial Loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN"

    # ==========================================
    # 4. Training Loop Execution
    # ==========================================
    print("\n>>> [4/6] Executing Training Loop (Trainer)...")

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    checkpoint_path = os.path.join(Config.WORKING_DIR, "checkpoint.pth")

    trainer = Trainer(
        model=model,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        patience=5,
        checkpoint_path=checkpoint_path,
    )

    # Run training for defined epochs
    trainer.fit(train_loader, val_loader, epochs=Config.EPOCHS)

    assert os.path.exists(checkpoint_path), "Checkpoint file was not created"
    print("    Training complete. Checkpoint saved.")

    # ==========================================
    # 5. Inference
    # ==========================================
    print("\n>>> [5/6] Running Inference on Test Set...")

    # Load best model
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            X_test = batch["X"].to(device)
            p = model(X_test)
            predictions.append(p.cpu().numpy().flatten())

    all_preds = np.concatenate(predictions)
    print(f"    Generated {len(all_preds)} predictions.")

    # ==========================================
    # 6. Submission Generation
    # ==========================================
    print("\n>>> [6/6] Formatting Submission...")

    # Load metadata
    test_meta = pd.read_csv(Config.TEST_METADATA)

    # In Debug mode, we only have predictions for a subset of breaths.
    # The DataFactory sorts data by breath_id and time_step.
    # We must align metadata to this order and slice it to match prediction count.

    # 1. Sort metadata to ensure alignment with DataFactory output
    # Note: 'id' correlates with 'time_step' in this dataset.
    test_meta_sorted = test_meta.sort_values(["breath_id", "id"]).reset_index(drop=True)

    # 2. Slice metadata to match the number of predictions (Debug Subset)
    # In full run, len(all_preds) == len(test_meta)
    submission_subset = test_meta_sorted.iloc[: len(all_preds)].copy()

    # 3. Assign predictions
    submission_subset["pressure"] = all_preds

    # 4. Save
    submission_path = os.path.join(Config.WORKING_DIR, "submission.csv")
    final_submission = submission_subset[["id", "pressure"]]
    final_submission.to_csv(submission_path, index=False)

    print(f"    Submission saved to: {submission_path}")

    # Verification
    df_check = pd.read_csv(submission_path)
    assert len(df_check) == len(all_preds), "Submission length mismatch"
    assert (
        "id" in df_check.columns and "pressure" in df_check.columns
    ), "Submission columns missing"

    print("\n>>> Demo Execution Completed Successfully.")


if __name__ == "__main__":
    main()
