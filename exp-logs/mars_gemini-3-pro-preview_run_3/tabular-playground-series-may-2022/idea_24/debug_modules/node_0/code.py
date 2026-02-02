import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.data_processing import preprocess_features, ManufacturingDataset
from library.model import ParallelFunnelEnsemble
from library.training import train_epoch, validate, train_model, predict_and_submit


def run_demonstration():
    print("==================================================")
    print("      Manufacturing Control - Code Demonstration  ")
    print("==================================================")

    # ---------------------------------------------------------
    # 1. Configuration Override for Speed and Isolation
    # ---------------------------------------------------------
    print("\n[1] Configuring Environment for Demo...")

    # Enable Debug mode to use a small subset of data
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 2000  # Use only 2000 samples for speed

    # Reduce training duration
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 128  # Smaller batch size for the small dataset

    # Redirect outputs to a specific demo directory to avoid overwriting production files
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = "./working/demo_execution"
    Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Manually update dependent paths since they are class attributes initialized at import
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    Config.CACHE_TRAIN_PATH = os.path.join(
        Config.WORKING_DIR, "train_processed.parquet"
    )
    Config.CACHE_VAL_PATH = os.path.join(Config.WORKING_DIR, "val_processed.parquet")
    Config.CACHE_TEST_PATH = os.path.join(Config.WORKING_DIR, "test_processed.parquet")
    Config.CACHE_META_PATH = os.path.join(Config.WORKING_DIR, "metadata.npy")
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")

    # Clean up any previous demo metadata to ensure a fresh run
    if os.path.exists(Config.CACHE_META_PATH):
        os.remove(Config.CACHE_META_PATH)

    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Debug Mode: {Config.DEBUG} (Size: {Config.DEBUG_SAMPLE_SIZE})")

    # ---------------------------------------------------------
    # 2. Data Processing Demonstration
    # ---------------------------------------------------------
    print("\n[2] Demonstrating Data Processing...")

    # Force reload=False (which implies processing from scratch if cache is deleted/invalid)
    # or explicitly set load_cached_data=False to force regeneration for the demo
    train_df, val_df, test_df, vocab_sizes, cat_cols, cont_cols = preprocess_features(
        load_cached_data=False, config=Config
    )

    print(f"    Processed Train Shape: {train_df.shape}")
    print(f"    Processed Val Shape:   {val_df.shape}")
    print(f"    Processed Test Shape:  {test_df.shape}")
    print(f"    Categorical Columns:   {len(cat_cols)}")
    print(f"    Continuous Columns:    {len(cont_cols)}")
    print(f"    Vocab Sizes:           {vocab_sizes}")

    # Validation
    assert len(train_df) == Config.DEBUG_SAMPLE_SIZE, "Train DF size mismatch"
    assert len(val_df) == Config.DEBUG_SAMPLE_SIZE, "Val DF size mismatch"
    assert len(test_df) == Config.DEBUG_SAMPLE_SIZE, "Test DF size mismatch"
    assert len(vocab_sizes) == len(cat_cols), "Vocab sizes list length mismatch"
    assert (
        "unique_char_count" in cont_cols
    ), "Feature engineering (f_27 decomposition) failed"

    # ---------------------------------------------------------
    # 3. Dataset Class Demonstration
    # ---------------------------------------------------------
    print("\n[3] Demonstrating Dataset Class...")

    train_dataset = ManufacturingDataset(train_df, cat_cols, cont_cols, mode="train")

    # Fetch a single sample
    x_cont_sample, x_cat_sample, y_sample = train_dataset[0]

    print(f"    x_cont shape: {x_cont_sample.shape} (dtype: {x_cont_sample.dtype})")
    print(f"    x_cat shape:  {x_cat_sample.shape} (dtype: {x_cat_sample.dtype})")
    print(f"    Target:       {y_sample} (dtype: {y_sample.dtype})")

    # Validation
    assert x_cont_sample.shape[0] == len(
        cont_cols
    ), "Continuous feature dimension mismatch"
    assert x_cat_sample.shape[0] == len(
        cat_cols
    ), "Categorical feature dimension mismatch"
    assert isinstance(y_sample, torch.Tensor), "Target is not a tensor"

    # ---------------------------------------------------------
    # 4. Model Architecture Demonstration
    # ---------------------------------------------------------
    print("\n[4] Demonstrating Model Architecture...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Device: {device}")

    model = ParallelFunnelEnsemble(
        vocab_sizes=vocab_sizes,
        cont_dim=len(cont_cols),
        embed_dim=Config.EMBED_DIM,
        stream_configs=Config.MODEL_STREAMS,
    ).to(device)

    # Create dummy batch
    dummy_batch_size = 4
    dummy_cont = torch.randn(dummy_batch_size, len(cont_cols)).to(device)
    dummy_cat = torch.randint(0, 2, (dummy_batch_size, len(cat_cols))).to(
        device
    )  # dummy indices

    # Forward Pass
    logits = model(dummy_cont, dummy_cat)
    print(f"    Output Logits Shape: {logits.shape}")

    # Validation
    # Expect shape: (batch_size, num_streams). There are 5 streams in Config.MODEL_STREAMS
    expected_streams = len(Config.MODEL_STREAMS)
    assert logits.shape == (
        dummy_batch_size,
        expected_streams,
    ), f"Model output shape mismatch. Expected ({dummy_batch_size}, {expected_streams}), got {logits.shape}"

    # ---------------------------------------------------------
    # 5. Training Loop Component Demonstration
    # ---------------------------------------------------------
    print("\n[5] Demonstrating Training Components (Epoch & Validation)...")

    # Setup DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_dataset = ManufacturingDataset(val_df, cat_cols, cont_cols, mode="val")
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Setup Optimizer/Scheduler/Loss
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-2, total_steps=len(train_loader) * Config.EPOCHS
    )
    criterion = nn.BCEWithLogitsLoss()

    # Run one training epoch
    train_loss = train_epoch(
        model, train_loader, optimizer, scheduler, criterion, device
    )
    print(f"    Single Epoch Train Loss: {train_loss:.6f}")
    assert not np.isnan(train_loss), "Training loss returned NaN"

    # Run validation
    val_auc = validate(model, val_loader, device)
    print(f"    Validation AUC: {val_auc:.6f}")
    assert 0.0 <= val_auc <= 1.0, "AUC score is out of valid range [0, 1]"

    # ---------------------------------------------------------
    # 6. Full Pipeline Integration
    # ---------------------------------------------------------
    print("\n[6] Demonstrating Full Pipeline (train_model & predict_and_submit)...")

    # Run the full training routine (will run for Config.EPOCHS=1)
    # This function handles data loading, model init, loop, and saving internally
    train_model()

    # Verify model file creation
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(
            f"Model checkpoint was not saved to {Config.MODEL_PATH}"
        )
    print(f"    Model saved successfully to: {Config.MODEL_PATH}")

    # Run prediction and submission generation
    predict_and_submit()

    # Verify submission file creation
    if not os.path.exists(Config.SUBMISSION_FILE):
        raise FileNotFoundError(
            f"Submission file was not created at {Config.SUBMISSION_FILE}"
        )

    # Check submission content
    sub_df = pd.read_csv(Config.SUBMISSION_FILE)
    print(f"    Submission File Shape: {sub_df.shape}")
    print(f"    Submission Head:\n{sub_df.head(3)}")

    assert (
        len(sub_df) == Config.DEBUG_SAMPLE_SIZE
    ), f"Submission rows ({len(sub_df)}) do not match debug test size ({Config.DEBUG_SAMPLE_SIZE})"

    print("\n==================================================")
    print("      Demonstration Completed Successfully        ")
    print("==================================================")


if __name__ == "__main__":
    run_demonstration()
