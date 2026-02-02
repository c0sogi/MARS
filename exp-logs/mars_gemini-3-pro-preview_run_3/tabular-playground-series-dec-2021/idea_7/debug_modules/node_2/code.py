import sys
import os
import torch
import pandas as pd
import numpy as np

# Ensure the current directory is in the path to import the library modules
sys.path.append(os.getcwd())

from library.config import Config
from library.model import ParallelDCNResNet, run_training_pipeline
from library.data_loader import get_dataloaders
from library.train_eval import train_model


def main():
    print("Starting Parallel DCN-ResNet Demonstration...")

    # --------------------------------------------------------------------------
    # 1. Configuration Override for Speed
    # --------------------------------------------------------------------------
    print("Overriding Config parameters for rapid execution...")
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 256
    Config.MAX_DEBUG_SAMPLES = 2000  # Restrict to 2000 samples for training/val
    Config.NUM_WORKERS = 0  # Disable multiprocessing for small data overhead

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Clean up specific cache files to force the data processing logic to run at least once
    if os.path.exists(Config.CACHE_TRAIN_X):
        os.remove(Config.CACHE_TRAIN_X)

    # --------------------------------------------------------------------------
    # 2. Data Loader Demonstration
    # --------------------------------------------------------------------------
    print("\n[Step 1] Testing Data Loader (get_dataloaders)...")

    # We set load_cached=False to demonstrate the raw data processing pipeline
    train_loader, val_loader, test_loader, test_ids, class_map = get_dataloaders(
        load_cached=False,
        batch_size=Config.BATCH_SIZE,
        debug_samples=Config.MAX_DEBUG_SAMPLES,
    )

    # Validations
    print("Validating Data Loader outputs...")
    assert isinstance(
        train_loader, torch.utils.data.DataLoader
    ), "train_loader is not a DataLoader"
    assert isinstance(
        val_loader, torch.utils.data.DataLoader
    ), "val_loader is not a DataLoader"
    assert len(test_ids) > 0, "test_ids should not be empty"

    # Inspect a single batch
    sample_inputs, sample_targets = next(iter(train_loader))
    input_dim = sample_inputs.shape[1]
    num_classes = len(class_map)

    print(f"  Batch Shape: {sample_inputs.shape}")
    print(f"  Input Features: {input_dim}")
    print(f"  Number of Classes: {num_classes}")

    # Verify dimensions (Original 54 + 3 engineered = 57 features expected)
    assert input_dim >= 54, f"Expected at least 54 features, got {input_dim}"
    assert num_classes > 1, "Expected multiple classes"

    # --------------------------------------------------------------------------
    # 3. Model Architecture Demonstration
    # --------------------------------------------------------------------------
    print("\n[Step 2] Testing Model Architecture (ParallelDCNResNet)...")

    # Instantiate model with reduced complexity for demo
    model = ParallelDCNResNet(
        input_dim=input_dim, num_classes=num_classes, resnet_blocks=1, dcn_layers=1
    )
    model.to(Config.DEVICE)

    # Forward Pass Verification
    dummy_input = sample_inputs.to(Config.DEVICE)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"  Model Output Shape: {output.shape}")

    assert output.shape == (
        sample_inputs.shape[0],
        num_classes,
    ), f"Output shape mismatch. Expected {(sample_inputs.shape[0], num_classes)}, got {output.shape}"
    assert not torch.isnan(output).any(), "Model output contains NaNs"

    # --------------------------------------------------------------------------
    # 4. Training Loop Demonstration
    # --------------------------------------------------------------------------
    print("\n[Step 3] Testing Training Loop (train_model)...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    trained_model, best_val_acc = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=Config.EPOCHS,
        device=Config.DEVICE,
    )

    print(f"  Training finished. Best Validation Accuracy: {best_val_acc:.4f}")
    assert 0.0 <= best_val_acc <= 1.0, "Validation accuracy out of bounds"

    # --------------------------------------------------------------------------
    # 5. Full Pipeline Integration Demonstration
    # --------------------------------------------------------------------------
    print("\n[Step 4] Testing Full Pipeline Wrapper (run_training_pipeline)...")

    # This function handles the end-to-end process: Loading -> Training -> Inference -> Submission
    # We use load_cached=True to leverage the data processed in Step 1
    pipeline_acc = run_training_pipeline(
        epochs=1,  # Minimal epochs for integration test
        batch_size=Config.BATCH_SIZE,
        load_cached=True,
    )

    print(f"  Pipeline finished. Validation Accuracy: {pipeline_acc:.4f}")

    # Verify Submission File
    submission_path = Config.SUBMISSION_FILE
    print(f"  Verifying submission file at: {submission_path}")

    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file not found at {submission_path}")

    df_sub = pd.read_csv(submission_path)
    print(f"  Submission Shape: {df_sub.shape}")

    # Check Columns
    assert Config.ID_COL in df_sub.columns, f"Missing ID column {Config.ID_COL}"
    assert (
        Config.TARGET_COL in df_sub.columns
    ), f"Missing Target column {Config.TARGET_COL}"

    # Check Row Count (Should match full test set size, which is 400,000)
    # Note: get_data loads the full test set even when debug_samples is set for train/val.
    expected_test_size = 400000
    assert (
        len(df_sub) == expected_test_size
    ), f"Submission row count mismatch. Expected {expected_test_size}, got {len(df_sub)}"

    print("\nSUCCESS: All demonstrations and validations passed.")


if __name__ == "__main__":
    main()
