import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, get_device
from library.dataset import prepare_data
from library.model import DSPIN
from library.engine import fit, generate_submission

if __name__ == "__main__":
    # ==========================================
    # 1. Setup and Configuration Override
    # ==========================================
    print("Initializing Demo Script...")

    # Set seeds for reproducibility
    seed_everything(42)

    # Override Config for a fast demonstration
    Config.DEBUG = True
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 16
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Debug Mode: {Config.DEBUG}")

    # ==========================================
    # 2. Metadata Patching for Consistency
    # ==========================================
    # In debug mode, the data loader only loads 100 breaths.
    # However, generate_submission reads the full test_metadata file to map IDs.
    # We must create a subset of test_metadata that matches the debug data subset
    # so that len(predictions) == len(metadata).

    print("Patching Test Metadata for Debug consistency...")
    original_test_meta_path = "./metadata/test_metadata.csv"
    debug_test_meta_path = os.path.join(Config.WORKING_DIR, "test_metadata_debug.csv")

    df_test_meta = pd.read_csv(original_test_meta_path)
    # The debug logic in feature_engineering.py takes the first 100 unique breath_ids
    unique_breaths = df_test_meta[Config.BREATH_ID_COL].unique()
    debug_breaths = unique_breaths[:100]
    df_test_meta_debug = df_test_meta[
        df_test_meta[Config.BREATH_ID_COL].isin(debug_breaths)
    ].copy()

    df_test_meta_debug.to_csv(debug_test_meta_path, index=False)

    # Point Config to the debug metadata
    Config.TEST_METADATA = debug_test_meta_path
    print(f"Test Metadata patched: {len(df_test_meta_debug)} rows.")

    # ==========================================
    # 3. Data Preparation
    # ==========================================
    print("\n--- Step 3: Data Preparation ---")
    # We force reprocessing (load_cached_data=False) to demonstrate the pipeline
    train_ds, val_ds, test_ds, scaler = prepare_data(load_cached_data=False, debug=True)

    # Verification
    print(f"Train Dataset Size: {len(train_ds)}")
    print(f"Val Dataset Size: {len(val_ds)}")
    print(f"Test Dataset Size: {len(test_ds)}")

    assert len(train_ds) > 0, "Train dataset is empty."
    assert len(val_ds) > 0, "Validation dataset is empty."

    # Check shape: (80, Features)
    sample_x = train_ds[0]["x"]
    sample_y = train_ds[0]["y"]
    input_dim = sample_x.shape[-1]

    print(f"Input Feature Dimension: {input_dim}")
    assert (
        sample_x.shape[0] == 80
    ), f"Expected sequence length 80, got {sample_x.shape[0]}"
    assert sample_y.shape[0] == 80, "Target sequence length mismatch."

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing overhead in demo
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # ==========================================
    # 4. Model Initialization
    # ==========================================
    print("\n--- Step 4: Model Initialization ---")
    device = get_device()
    print(f"Device: {device}")

    model = DSPIN(input_dim=input_dim).to(device)

    # Verification: Dummy Forward Pass
    dummy_input = torch.randn(2, 80, input_dim).to(device)
    with torch.no_grad():
        dummy_out = model(dummy_input)

    print(f"Model Output Shape: {dummy_out.shape}")
    assert dummy_out.shape == (2, 80, 1), "Model output shape mismatch."

    # ==========================================
    # 5. Training Loop
    # ==========================================
    print("\n--- Step 5: Training ---")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=Config.EPOCHS,
    )

    # Verify Model Saved
    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model file was not saved."
    print("Model training completed and saved successfully.")

    # ==========================================
    # 6. Inference and Submission
    # ==========================================
    print("\n--- Step 6: Inference & Submission ---")

    # Load best model state
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    generate_submission(model, test_loader, device)

    # Verify Submission
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission Shape: {df_sub.shape}")
    print(f"Submission Columns: {df_sub.columns.tolist()}")

    assert (
        "id" in df_sub.columns and "pressure" in df_sub.columns
    ), "Submission columns missing."
    assert len(df_sub) == len(
        df_test_meta_debug
    ), "Submission length mismatch with metadata."

    print("\nDemo execution completed successfully.")
