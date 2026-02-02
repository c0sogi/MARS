import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

# Import library components
from library.config import Config
from library.data_processing import (
    set_seed,
    load_and_preprocess,
    get_dataloaders,
    feature_engineering,
)
from library.model import RSPFEModel
from library.training import train_one_epoch, evaluate


def main():
    print("Starting RSPFE Library Demonstration...")

    # ==========================================
    # 1. Configuration Override for Speed
    # ==========================================
    # We modify the Config class directly to enable debug mode and reduce runtime.
    print("Configuring environment for fast execution...")
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 2000  # Use a small subset of 2000 samples
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.BATCH_SIZE = 64  # Smaller batch size

    # Ensure reproducibility
    set_seed(Config.SEED)

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # ==========================================
    # 2. Data Processing & Verification
    # ==========================================
    print("Loading and preprocessing data (forcing scratch build for debug subset)...")
    # We set load_cached_data=False to ensure we process the small debug subset
    # instead of loading potentially large cached files from a previous run.
    df_train, df_val, df_test, vocab_sizes = load_and_preprocess(load_cached_data=False)

    print("Verifying Data Processing outputs...")
    # Verify dataset sizes
    assert (
        len(df_train) == Config.DEBUG_SAMPLES
    ), f"Train set size mismatch. Expected {Config.DEBUG_SAMPLES}, got {len(df_train)}"

    # Validation set size logic in library is min(len, debug//5)
    expected_val_size = min(160000, Config.DEBUG_SAMPLES // 5)
    assert (
        len(df_val) == expected_val_size
    ), f"Val set size mismatch. Expected {expected_val_size}, got {len(df_val)}"

    # Verify Feature Engineering
    # Check if 'unique_character_count' and decomposed characters exist
    assert (
        "unique_character_count" in df_train.columns
    ), "Feature engineering failed: unique_character_count missing"
    assert (
        "f_27_0" in df_train.columns
    ), "Feature engineering failed: f_27 decomposition missing"

    # Verify Vocab Sizes
    # Config.CAT_FEATURES has 12 items: f_27_0..9 (10) + f_29 (1) + f_30 (1)
    assert (
        len(vocab_sizes) == 12
    ), f"Expected 12 categorical features in vocab_sizes, got {len(vocab_sizes)}"

    # ==========================================
    # 3. DataLoader Verification
    # ==========================================
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        df_train, df_val, df_test, batch_size=Config.BATCH_SIZE
    )

    # Fetch a single batch to verify structure and shapes
    batch = next(iter(train_loader))
    cat_features = batch["cat_features"].to(device)
    cont_features = batch["cont_features"].to(device)
    targets = batch["target"].to(device)

    print("Verifying Batch Shapes...")
    # Categorical shape: (Batch, 12)
    assert cat_features.shape == (
        Config.BATCH_SIZE,
        12,
    ), f"Categorical features shape mismatch: {cat_features.shape}"

    # Continuous shape: 27 (f_00-f_26) + 1 (f_28) + 1 (unique_char_count) = 29
    assert cont_features.shape == (
        Config.BATCH_SIZE,
        29,
    ), f"Continuous features shape mismatch: {cont_features.shape}"

    # Target shape: (Batch,)
    assert targets.shape == (
        Config.BATCH_SIZE,
    ), f"Target shape mismatch: {targets.shape}"

    # ==========================================
    # 4. Model Initialization & Forward Pass
    # ==========================================
    print("Initializing RSPFE Model...")
    model = RSPFEModel(vocab_sizes=vocab_sizes)
    model.to(device)

    print("Verifying Model Forward Pass...")
    with torch.no_grad():
        # Forward pass
        outputs = model(cat_features, cont_features)

    # The model should output 5 logits (one for each stream) per sample
    assert outputs.shape == (
        Config.BATCH_SIZE,
        5,
    ), f"Model output shape mismatch. Expected ({Config.BATCH_SIZE}, 5), got {outputs.shape}"

    print("Model initialized and forward pass successful.")

    # ==========================================
    # 5. Training Loop Demonstration
    # ==========================================
    print("Demonstrating Training Loop...")

    # Setup Optimizer and Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=Config.EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=Config.PCT_START,
        div_factor=Config.DIV_FACTOR,
        final_div_factor=Config.FINAL_DIV_FACTOR,
    )
    criterion = nn.BCEWithLogitsLoss()

    # Run one epoch of training
    train_loss = train_one_epoch(
        model, train_loader, criterion, optimizer, scheduler, device
    )
    print(f"Training Epoch Completed. Loss: {train_loss:.4f}")

    # Verify Loss is valid
    assert not np.isnan(train_loss), "Training loss is NaN"
    assert train_loss > 0, "Training loss should be positive"

    # ==========================================
    # 6. Evaluation & Inference Demonstration
    # ==========================================
    print("Demonstrating Evaluation...")
    val_auc, val_preds = evaluate(model, val_loader, device)
    print(f"Validation AUC: {val_auc:.4f}")

    # Verify AUC range
    assert 0.0 <= val_auc <= 1.0, f"AUC score {val_auc} is out of bounds [0, 1]"
    assert len(val_preds) == len(df_val), "Mismatch in number of validation predictions"

    print("Demonstrating Inference on Test Set...")
    model.eval()
    test_preds = []
    with torch.no_grad():
        for batch in test_loader:
            c_cat = batch["cat_features"].to(device)
            c_cont = batch["cont_features"].to(device)

            # Forward
            logits = model(c_cat, c_cont)
            probs = torch.sigmoid(logits)

            # Mean across 5 streams
            mean_preds = probs.mean(dim=1)
            test_preds.extend(mean_preds.cpu().numpy())

    assert len(test_preds) == len(
        df_test
    ), f"Test predictions count mismatch. Expected {len(df_test)}, got {len(test_preds)}"

    print("All demonstrations and verifications passed successfully.")


if __name__ == "__main__":
    main()
