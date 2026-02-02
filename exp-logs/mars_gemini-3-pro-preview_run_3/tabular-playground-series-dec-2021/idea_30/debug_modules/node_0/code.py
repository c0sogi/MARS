import os
import sys
import torch
import pandas as pd
import numpy as np

# Import provided library modules
import library.config
import library.utils
import library.data_loader
import library.model
import library.trainer


def main():
    print("=== Starting Library Demonstration ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Patching
    # -------------------------------------------------------------------------
    # We override default configurations to ensure the demo runs quickly (Speed Optimization).
    # We enable DEBUG mode to use a small subset of data and reduce training epochs.
    print("\n[1] Patching configuration for fast execution...")

    # Patch library.config (The central config)
    library.config.DEBUG = True
    library.config.DEBUG_SUBSET_SIZE = 2000  # Use only 2000 rows
    library.config.EPOCHS = 1  # Train for only 1 epoch
    library.config.BATCH_SIZE = 128  # Smaller batch size

    # Patch library.data_loader (Variables imported via 'from ... import ...')
    library.data_loader.DEBUG = True
    library.data_loader.DEBUG_SUBSET_SIZE = 2000
    library.data_loader.BATCH_SIZE = 128

    # Patch library.model (Variables imported via 'from ... import ...')
    library.model.EPOCHS = 1
    library.model.HIDDEN_DIM = 64  # Reduce model size for speed

    # Patch library.trainer (Variables imported via 'from ... import ...')
    library.trainer.EPOCHS = 1

    print("    Configuration patched: DEBUG=True, EPOCHS=1, SUBSET=2000")

    # -------------------------------------------------------------------------
    # 2. Reproducibility
    # -------------------------------------------------------------------------
    print("\n[2] Setting random seeds...")
    library.utils.seed_everything(42)

    # -------------------------------------------------------------------------
    # 3. Data Loading
    # -------------------------------------------------------------------------
    print("\n[3] Demonstrating Data Loading...")

    # We set load_cached_data=False to force the processing logic to run on the debug subset
    train_loader, val_loader, test_loader, test_ids = (
        library.data_loader.get_dataloaders(
            load_cached_data=False, batch_size=library.config.BATCH_SIZE
        )
    )

    # Verify DataLoaders
    print(f"    Train Batches: {len(train_loader)}")
    print(f"    Val Batches:   {len(val_loader)}")
    print(f"    Test Batches:  {len(test_loader)}")
    print(f"    Test IDs:      {len(test_ids)}")

    # Assertions to verify logic
    assert len(train_loader) > 0, "Train loader should not be empty."
    assert len(val_loader) > 0, "Validation loader should not be empty."
    assert len(test_loader) > 0, "Test loader should not be empty."
    assert (
        len(test_ids) == library.config.DEBUG_SUBSET_SIZE
    ), "Test IDs count mismatch with debug subset."

    # Inspect a single batch
    X_batch, y_batch = next(iter(train_loader))
    print(f"    Sample Batch X: {X_batch.shape}")
    print(f"    Sample Batch y: {y_batch.shape}")

    input_dim = X_batch.shape[1]
    num_classes = library.config.NUM_CLASSES

    # -------------------------------------------------------------------------
    # 4. Model Initialization & Forward Pass
    # -------------------------------------------------------------------------
    print("\n[4] Demonstrating Model Architecture...")

    device = library.utils.get_device()
    print(f"    Device: {device}")

    # Instantiate the model
    model = library.model.ParallelDCNResNet(
        input_dim=input_dim,
        hidden_dim=library.model.HIDDEN_DIM,
        num_classes=num_classes,
        dropout=0.1,
    ).to(device)

    # Verify Forward Pass
    X_batch = X_batch.to(device)
    with torch.no_grad():
        outputs = model(X_batch)

    print(f"    Model Output Shape: {outputs.shape}")

    # Assert output shape is (Batch_Size, Num_Classes)
    assert outputs.shape == (
        X_batch.size(0),
        num_classes,
    ), f"Expected output shape {(X_batch.size(0), num_classes)}, got {outputs.shape}"

    # -------------------------------------------------------------------------
    # 5. Training Loop
    # -------------------------------------------------------------------------
    print("\n[5] Demonstrating Training Loop (1 Epoch)...")

    # Instantiate Trainer
    trainer = library.trainer.Trainer(model, train_loader, val_loader, device)

    # Run Training
    trainer.fit(epochs=library.trainer.EPOCHS)

    # Verify Model Artifacts
    model_path = library.config.MODEL_PATH
    if os.path.exists(model_path):
        print(f"    Model successfully saved to: {model_path}")
    else:
        raise AssertionError(f"Model file not found at {model_path}")

    # -------------------------------------------------------------------------
    # 6. Prediction & Submission
    # -------------------------------------------------------------------------
    print("\n[6] Demonstrating Prediction pipeline...")

    # Generate Predictions
    library.trainer.generate_predictions(model, test_loader, test_ids, device)

    # Verify Submission File
    sub_path = library.config.SUBMISSION_PATH
    if os.path.exists(sub_path):
        print(f"    Submission file created at: {sub_path}")

        # Load and validate content
        df_sub = pd.read_csv(sub_path)
        print(f"    Submission Shape: {df_sub.shape}")

        assert (
            df_sub.shape[0] == library.config.DEBUG_SUBSET_SIZE
        ), "Submission row count does not match test set size."
        assert list(df_sub.columns) == [
            "Id",
            "Cover_Type",
        ], "Submission columns are incorrect."
        assert not df_sub.isnull().values.any(), "Submission contains NaN values."

        print("    Submission file content valid.")
    else:
        raise AssertionError(f"Submission file not found at {sub_path}")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
