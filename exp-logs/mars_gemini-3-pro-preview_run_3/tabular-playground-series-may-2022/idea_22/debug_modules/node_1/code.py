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
from library.data_utils import load_and_preprocess_data, TabularDataset
from library.model import AVPFE
from library.train_utils import train_epoch, validate


def set_seed(seed=42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def demo_pipeline():
    print("Starting Demo Pipeline...")

    # 1. Configure for Demo (Speed Optimization)
    # Redirect working directories to a demo folder to avoid conflicts
    DEMO_DIR = "./working/demo_run"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    print(f"Setting Config paths to {DEMO_DIR}")
    Config.WORKING_DIR = DEMO_DIR
    Config.CACHE_DIR = DEMO_DIR
    Config.SUBMISSION_DIR = DEMO_DIR
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission.csv")

    # Reduce training parameters for rapid execution
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 128  # Smaller batch for the small sliced dataset

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Config.DEVICE = device
    print(f"Using device: {device}")

    # 2. Load Data
    # This will process metadata/train.csv etc. and save parquets to DEMO_DIR
    # We force reload to ensure the demo is self-contained and uses the new directory
    print("Loading and preprocessing data...")
    train_ds_full, val_ds_full, test_ds_full = load_and_preprocess_data(
        load_cached_data=False
    )

    # Verify Data Types
    assert isinstance(
        train_ds_full, TabularDataset
    ), "Train dataset is not TabularDataset"
    assert isinstance(val_ds_full, TabularDataset), "Val dataset is not TabularDataset"

    # 3. Slice Data for Speed
    # We take a small subset (e.g., 1000 samples) to ensure the demo runs instantly
    # while still exercising the training loop.
    SUBSET_SIZE = 1000
    print(f"Slicing datasets to {SUBSET_SIZE} samples for speed verification...")

    def slice_dataset(ds, n, has_target=True):
        # Slice tensors and create a new Dataset instance
        x_cat = ds.x_cat[:n]
        x_cont = ds.x_cont[:n]
        y = ds.y[:n] if has_target else None

        # Convert back to numpy for the constructor (which expects arrays)
        # Fix: Squeeze y (remove dim 1) because TabularDataset adds it back
        y_np = y.squeeze(1).numpy() if y is not None else None

        return TabularDataset(x_cat.numpy(), x_cont.numpy(), y_np)

    train_ds = slice_dataset(train_ds_full, SUBSET_SIZE, has_target=True)
    val_ds = slice_dataset(val_ds_full, SUBSET_SIZE, has_target=True)
    test_ds = slice_dataset(test_ds_full, SUBSET_SIZE, has_target=False)

    # Verify Slicing
    assert len(train_ds) == SUBSET_SIZE
    assert len(val_ds) == SUBSET_SIZE
    assert len(test_ds) == SUBSET_SIZE

    # 4. Initialize Model
    # We need to calculate cardinalities based on the FULL dataset to ensure embeddings
    # are sized correctly for all potential values, even if our slice doesn't see them all.
    # Concatenate full x_cat to find max indices.
    full_cat = torch.cat(
        [train_ds_full.x_cat, val_ds_full.x_cat, test_ds_full.x_cat], dim=0
    )
    cat_cardinalities = (full_cat.max(dim=0).values + 1).tolist()
    n_cont = train_ds.x_cont.shape[1]

    print(
        f"Initializing AVPFE model with {len(cat_cardinalities)} categorical features and {n_cont} continuous features."
    )
    model = AVPFE(cat_cardinalities, n_cont).to(device)

    # Verify Model Structure
    # Check if we have 5 streams as defined in Config
    assert len(model.streams) == 5, "Model should have 5 streams defined in Config"

    # 5. Test Forward Pass
    print("Testing forward pass...")
    # Create a dummy batch
    dummy_loader = DataLoader(train_ds, batch_size=4, shuffle=False)
    x_cat_batch, x_cont_batch, y_batch = next(iter(dummy_loader))
    x_cat_batch, x_cont_batch = x_cat_batch.to(device), x_cont_batch.to(device)

    with torch.no_grad():
        outputs = model(x_cat_batch, x_cont_batch)

    # Output shape should be (Batch_Size, Num_Streams) -> (4, 5)
    assert outputs.shape == (4, 5), f"Expected output shape (4, 5), got {outputs.shape}"
    print("Forward pass successful.")

    # 6. Training Loop Demonstration
    print("Starting training loop demonstration...")
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.MAX_LR, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Scheduler setup (OneCycleLR requires total steps)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.MAX_LR,
        epochs=Config.EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
    )

    for epoch in range(Config.EPOCHS):
        avg_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, criterion
        )
        val_loss, val_auc = validate(model, val_loader, device, criterion)

        print(
            f"Epoch {epoch+1}: Train Loss={avg_loss:.4f}, Val Loss={val_loss:.4f}, Val AUC={val_auc:.4f}"
        )

        # Assertions to ensure learning mechanics are working
        assert not np.isnan(avg_loss), "Training loss is NaN"
        assert not np.isnan(val_loss), "Validation loss is NaN"
        assert 0.0 <= val_auc <= 1.0, "AUC score out of range"

    # 7. Inference Demonstration
    print("Running inference on test set...")
    test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False)
    model.eval()
    predictions = []

    with torch.no_grad():
        for x_cat, x_cont in test_loader:
            x_cat, x_cont = x_cat.to(device), x_cont.to(device)
            outputs = model(x_cat, x_cont)
            # Ensemble mean probability
            probs = torch.sigmoid(outputs).mean(dim=1)
            predictions.extend(probs.cpu().numpy())

    assert len(predictions) == len(
        test_ds
    ), "Number of predictions does not match test set size"
    print(f"Generated {len(predictions)} predictions.")

    # 8. Submission File Generation
    # We create a dummy dataframe for the sliced test set to verify CSV writing
    print("Generating submission file...")
    # Get IDs corresponding to the slice from the metadata file
    df_test_full = pd.read_csv(Config.TEST_METADATA_PATH)
    df_test_slice = df_test_full.iloc[:SUBSET_SIZE].copy()

    df_test_slice["target"] = predictions
    # Keep only id and target
    submission_df = df_test_slice[["id", "target"]]

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    # Verify file existence
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"

    # Verify content format
    saved_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert list(saved_df.columns) == ["id", "target"], "Submission columns mismatch"
    assert len(saved_df) == SUBSET_SIZE, "Submission row count mismatch"

    print("Demo Pipeline Completed Successfully.")


if __name__ == "__main__":
    set_seed(42)
    demo_pipeline()
