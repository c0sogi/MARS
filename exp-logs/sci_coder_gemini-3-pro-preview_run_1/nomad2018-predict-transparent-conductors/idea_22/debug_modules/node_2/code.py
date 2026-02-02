import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from unittest.mock import patch

# Import library modules
from library.config import Config
from library.utils import seed_everything, get_device, rmsle, StandardScaler
from library.preprocessing import prepare_data
from library.data import get_dataloaders, collate_crystals
from library.model import MSNWDSModel
from library.engine import (
    EarlyStopping,
    train_one_epoch,
    validate,
    generate_predictions,
)
import library.preprocessing  # For monkeypatching


def main():
    print("Initializing demonstration...")

    # 1. Setup and Reproducibility
    seed_everything(Config.SEED)
    device = get_device()
    print(f"Device: {device}")

    # 2. Optimize for Speed: Monkey-patch pandas.read_csv in preprocessing
    # to only load a small subset of data. This avoids processing thousands of files.
    original_read_csv = library.preprocessing.pd.read_csv

    def fast_read_csv(*args, **kwargs):
        df = original_read_csv(*args, **kwargs)
        # Return only top 20 samples for speed
        if len(df) > 20:
            return df.head(20)
        return df

    print("Patching data loader for fast execution...")
    library.preprocessing.pd.read_csv = fast_read_csv

    # Force re-computation to use the small subset
    # We clear the cache files if they exist to ensure we run the extraction logic
    cache_files = ["train_data.npz", "val_data.npz", "test_data.npz", "scalers.npz"]
    for f in cache_files:
        p = os.path.join(Config.WORKING_DIR, f)
        if os.path.exists(p):
            os.remove(p)

    try:
        # 3. Data Loading and Processing
        print("\n--- Testing Data Pipeline ---")
        # Use a small batch size for the small dataset
        batch_size = 4
        train_loader, val_loader, test_loader = get_dataloaders(
            batch_size=batch_size, load_cached=False
        )

        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")

        assert len(train_loader) > 0, "Train loader is empty"
        assert len(val_loader) > 0, "Val loader is empty"

        # 4. Inspect a Batch
        print("\n--- Inspecting Batch Structure ---")
        batch = next(iter(train_loader))

        atomic = batch["atomic"]
        glob = batch["global"]
        mask = batch["mask"]
        target = batch["target"]
        ids = batch["id"]

        print(f"Atomic shape: {atomic.shape} (Batch, Max_Atoms, Feat_Dim)")
        print(f"Global shape: {glob.shape} (Batch, Feat_Dim)")
        print(f"Mask shape: {mask.shape} (Batch, Max_Atoms)")
        print(f"Target shape: {target.shape} (Batch, Targets)")

        # Assertions for shapes
        assert atomic.dim() == 3
        assert atomic.shape[2] == Config.ATOMIC_INPUT_DIM
        assert glob.dim() == 2
        assert glob.shape[1] == Config.GLOBAL_INPUT_DIM
        assert target.dim() == 2
        assert target.shape[1] == 2  # 2 targets
        assert mask.dim() == 2
        assert len(ids) == atomic.shape[0]

        # 5. Model Initialization and Forward Pass
        print("\n--- Testing Model ---")
        model = MSNWDSModel().to(device)

        # Move batch to device
        atomic = atomic.to(device)
        glob = glob.to(device)
        mask = mask.to(device)

        # Forward pass
        output = model(atomic, glob, mask)
        print(f"Model output shape: {output.shape}")

        assert output.shape == (atomic.shape[0], 2), "Model output shape mismatch"
        assert not torch.isnan(output).any(), "Model produced NaNs"

        # 6. Training Step
        print("\n--- Testing Training Step ---")
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Train One Epoch Loss: {loss:.6f}")
        assert loss > 0, "Training loss should be positive"

        # 7. Validation Step
        print("\n--- Testing Validation Step ---")
        val_loss, rmsle_form, rmsle_band = validate(
            model, val_loader, criterion, device
        )
        print(f"Val Loss: {val_loss:.6f}")
        print(f"RMSLE Formation: {rmsle_form:.6f}")
        print(f"RMSLE Bandgap: {rmsle_band:.6f}")

        assert val_loss >= 0, "Validation loss should be non-negative"

        # 8. Early Stopping Logic
        print("\n--- Testing Early Stopping ---")
        early_stopping = EarlyStopping(patience=2, verbose=True)

        # Simulate loss history: 0.5 -> 0.4 (improve) -> 0.45 (worse) -> 0.46 (worse, stop)
        print("Step 1: Loss 0.5")
        early_stopping(0.5, model)
        assert not early_stopping.early_stop

        print("Step 2: Loss 0.4")
        early_stopping(0.4, model)
        assert not early_stopping.early_stop
        assert early_stopping.best_score == -0.4

        print("Step 3: Loss 0.45")
        early_stopping(0.45, model)
        assert not early_stopping.early_stop
        assert early_stopping.counter == 1

        print("Step 4: Loss 0.46")
        early_stopping(0.46, model)
        assert early_stopping.early_stop
        print("Early stopping triggered correctly.")

        # 9. Inference / Submission Generation
        print("\n--- Testing Inference ---")
        # We use the generate_predictions function from engine.py
        # Using a temporary output path
        output_csv = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
        df_submission = generate_predictions(
            model, test_loader, device, output_path=output_csv
        )

        print("Submission DataFrame head:")
        print(df_submission.head())

        assert "id" in df_submission.columns
        assert "formation_energy_ev_natom" in df_submission.columns
        assert "bandgap_energy_ev" in df_submission.columns
        assert (
            len(df_submission) == 20
        ), f"Expected 20 predictions (due to mocking), got {len(df_submission)}"
        assert os.path.exists(output_csv), "Submission file was not created"

        # 10. Utils Verification
        print("\n--- Testing Utils ---")
        # StandardScaler
        scaler = StandardScaler()
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        scaler.fit(data)
        transformed = scaler.transform(data)
        inverse = scaler.inverse_transform(transformed)

        print(f"Original: \n{data}")
        print(f"Transformed: \n{transformed}")
        print(f"Inverse: \n{inverse}")

        assert np.allclose(data, inverse), "StandardScaler inverse transform failed"
        assert np.allclose(
            np.mean(transformed, axis=0), 0, atol=1e-7
        ), "StandardScaler mean not 0"

        # RMSLE
        y_true = np.array([1.0, 10.0, 100.0])
        y_pred = np.array([1.1, 9.5, 105.0])
        score = rmsle(y_true, y_pred)
        print(f"RMSLE Score: {score:.6f}")
        assert score >= 0, "RMSLE cannot be negative"

    finally:
        # Restore original read_csv to avoid side effects if this were part of a larger system
        library.preprocessing.pd.read_csv = original_read_csv
        print("\nRestored original pandas read_csv.")

    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    main()
