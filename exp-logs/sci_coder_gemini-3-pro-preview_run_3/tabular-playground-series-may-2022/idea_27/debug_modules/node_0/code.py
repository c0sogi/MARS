import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import provided library modules
from library.utils import seed_everything
from library.config import STREAM_CONFIGS
from library.preprocessing import ManufacturingPreprocessor
from library.dataset import ManufacturingDataset
from library.model import IAPEModel
from library.engine import train_model, generate_submission

# Define constants for the demo
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/demo_execution"
SAMPLE_SIZE_TRAIN = 2000
SAMPLE_SIZE_VAL = 500
SAMPLE_SIZE_TEST = 500
BATCH_SIZE = 32
EPOCHS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    print("Starting Demo Script...")

    # 1. Setup
    seed_everything(42)
    os.makedirs(WORKING_DIR, exist_ok=True)

    # 2. Data Loading (Optimized for Speed)
    print("Loading sampled data from metadata...")
    train_path = os.path.join(METADATA_DIR, "train.csv")
    val_path = os.path.join(METADATA_DIR, "val.csv")
    test_path = os.path.join(METADATA_DIR, "test.csv")

    # Load only a subset of rows to ensure quick execution
    df_train = pd.read_csv(train_path, nrows=SAMPLE_SIZE_TRAIN)
    df_val = pd.read_csv(val_path, nrows=SAMPLE_SIZE_VAL)
    df_test = pd.read_csv(test_path, nrows=SAMPLE_SIZE_TEST)

    print(f"Train shape: {df_train.shape}")
    print(f"Val shape: {df_val.shape}")
    print(f"Test shape: {df_test.shape}")

    # 3. Preprocessing
    print("Initializing and fitting Preprocessor...")
    preprocessor = ManufacturingPreprocessor()

    # Fit on the subsets (Transductive setting handled internally)
    preprocessor.fit(df_train, df_val, df_test)

    # Transform data
    train_proc = preprocessor.transform(df_train)
    val_proc = preprocessor.transform(df_val)
    test_proc = preprocessor.transform(df_test)

    # Verification: Check if engineered columns exist
    expected_new_cols = ["unique_char_count", "ch_0", "ch_9"]
    for col in expected_new_cols:
        if col not in train_proc.columns:
            raise AssertionError(f"Preprocessing failed: Column {col} missing.")

    print("Preprocessing complete.")

    # 4. Dataset & DataLoader
    print("Creating Datasets and DataLoaders...")
    cont_cols = preprocessor.cont_cols
    cat_cols = preprocessor.cat_cols

    train_ds = ManufacturingDataset(
        train_proc, cont_cols, cat_cols, target_col="target"
    )
    val_ds = ManufacturingDataset(val_proc, cont_cols, cat_cols, target_col="target")
    test_ds = ManufacturingDataset(test_proc, cont_cols, cat_cols, target_col=None)

    # Verification: Check single item structure
    sample_item = train_ds[0]
    assert "continuous" in sample_item
    assert "categorical" in sample_item
    assert "target" in sample_item
    assert isinstance(sample_item["continuous"], torch.Tensor)
    assert sample_item["continuous"].dtype == torch.float32
    assert sample_item["categorical"].dtype == torch.int64

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"DataLoaders created. Batch size: {BATCH_SIZE}")

    # 5. Model Initialization
    print("Initializing IAPEModel...")
    model = IAPEModel(
        num_cont=len(cont_cols),
        cat_cardinalities=preprocessor.cat_cardinalities,
        stream_configs=STREAM_CONFIGS,
    ).to(DEVICE)

    # Verification: Dummy forward pass
    dummy_cont = sample_item["continuous"].unsqueeze(0).to(DEVICE)
    dummy_cat = sample_item["categorical"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(dummy_cont, dummy_cat)

    # Output shape should be [batch_size, num_streams]
    # STREAM_CONFIGS has 5 entries
    expected_streams = len(STREAM_CONFIGS)
    assert output.shape == (
        1,
        expected_streams,
    ), f"Model output shape mismatch. Expected (1, {expected_streams}), got {output.shape}"

    print("Model initialized and verified.")

    # 6. Training
    print(f"Starting training for {EPOCHS} epochs on {DEVICE}...")

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    # Simple scheduler for demo
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, epochs=EPOCHS, steps_per_epoch=len(train_loader)
    )

    save_path = os.path.join(WORKING_DIR, "best_model_demo.pth")

    trained_model, best_auc = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=DEVICE,
        epochs=EPOCHS,
        patience=1,  # Strict patience for demo
        save_path=save_path,
    )

    print(f"Training complete. Best Val AUC: {best_auc:.4f}")

    # Verification: Check if model file was saved
    if not os.path.exists(save_path):
        raise AssertionError("Model checkpoint was not saved.")

    # 7. Submission Generation
    print("Generating submission...")
    submission_path = os.path.join(WORKING_DIR, "submission_demo.csv")
    test_ids = df_test["id"].values

    generate_submission(
        model=trained_model,
        test_loader=test_loader,
        test_ids=test_ids,
        device=DEVICE,
        output_path=submission_path,
    )

    # Verification: Check submission file
    if not os.path.exists(submission_path):
        raise AssertionError("Submission file not created.")

    sub_df = pd.read_csv(submission_path)
    assert sub_df.shape == (
        SAMPLE_SIZE_TEST,
        2,
    ), f"Submission shape mismatch. Expected ({SAMPLE_SIZE_TEST}, 2), got {sub_df.shape}"
    assert list(sub_df.columns) == ["id", "target"], "Submission columns mismatch."

    print(f"Submission generated at {submission_path}")
    print("Demo execution completed successfully.")


if __name__ == "__main__":
    main()
