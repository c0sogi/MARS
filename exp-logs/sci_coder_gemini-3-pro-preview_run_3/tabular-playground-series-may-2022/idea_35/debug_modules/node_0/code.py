import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

# Import provided library modules
import library.config as config
import library.data_loader as data_loader
import library.model as model_lib
import library.trainer as trainer_lib


def run_demo():
    print("Starting DSR-PE Model Demonstration...")

    # 1. Configuration Overrides for Speed
    # We modify the global config variables to make the demo run fast
    print("Configuring hyperparameters for fast demonstration...")
    config.EPOCHS = 2
    config.BATCH_SIZE = 128  # Smaller batch size for the demo subset

    # Set seeds
    trainer_lib.set_seed(config.SEED)

    # 2. Data Loading
    print("\n--- Step 1: Data Loading & Processing ---")
    # We force load_cached_data=False to demonstrate the feature engineering logic,
    # but in a real scenario with limited time, one might use True.
    # Given the constraints, we'll try to load cache if available to save time,
    # else process from scratch.
    train_ds, val_ds, test_ds, vocab_sizes = data_loader.prepare_data(
        load_cached_data=True
    )

    # Verification of Data Loading
    print("Verifying data structures...")
    assert isinstance(vocab_sizes, np.ndarray), "vocab_sizes should be a numpy array"
    assert len(vocab_sizes) > 0, "vocab_sizes should not be empty"
    assert isinstance(
        train_ds, data_loader.ManufacturingDataset
    ), "train_ds should be a ManufacturingDataset"

    # Check a single sample
    cat_sample, cont_sample, target_sample = train_ds[0]
    cont_dim = cont_sample.shape[0]
    print(f"Continuous Feature Dimension: {cont_dim}")
    print(f"Categorical Feature Count: {len(vocab_sizes)}")

    assert cont_sample.dtype == torch.float32, "Continuous features must be float32"
    assert cat_sample.dtype == torch.long, "Categorical features must be long"

    # 3. Create Subsets for Speed
    # We will use only 1000 samples for training and validation loops
    print("\n--- Step 2: Creating Subsets for Speed ---")
    subset_indices = range(1000)
    train_subset = Subset(train_ds, subset_indices)
    val_subset = Subset(val_ds, subset_indices)
    test_subset = Subset(test_ds, subset_indices)

    train_loader = DataLoader(train_subset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=config.BATCH_SIZE, shuffle=False)

    print(f"Train subset size: {len(train_subset)}")
    print(f"Val subset size: {len(val_subset)}")

    # 4. Model Instantiation & Logic Verification
    print("\n--- Step 3: Model Instantiation & Verification ---")
    model = model_lib.DSRPEModel(vocab_sizes=vocab_sizes, cont_dim=cont_dim)
    model.to(config.DEVICE)

    # Create a dummy batch to verify forward pass
    dummy_cat = torch.randint(0, 2, (config.BATCH_SIZE, len(vocab_sizes))).to(
        config.DEVICE
    )
    dummy_cont = torch.randn(config.BATCH_SIZE, cont_dim).to(config.DEVICE)

    print("Running forward pass check...")
    outputs = model(dummy_cat, dummy_cont)

    # Verify Model Output
    # The model should return a list of tensors (one per stream)
    assert isinstance(outputs, list), "Model output should be a list"
    assert len(outputs) == len(
        config.STREAM_CONFIGS
    ), f"Model should return {len(config.STREAM_CONFIGS)} stream outputs"

    for i, out in enumerate(outputs):
        assert out.shape == (
            config.BATCH_SIZE,
            1,
        ), f"Stream {i} output shape mismatch. Expected {(config.BATCH_SIZE, 1)}, got {out.shape}"

    print("Model forward pass verified successfully.")

    # 5. Training Loop Demonstration
    print("\n--- Step 4: Training Loop Demonstration ---")

    # Setup Optimizer, Scheduler, Criterion
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.LEARNING_RATE,
        epochs=config.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
    )

    # Run Training for 1 Epoch
    print("Running training for 1 epoch on subset...")
    train_loss = trainer_lib.train_one_epoch(
        model, train_loader, optimizer, scheduler, criterion, config.DEVICE
    )
    print(f"Train Loss: {train_loss:.4f}")
    assert not np.isnan(train_loss), "Training loss is NaN"
    assert train_loss > 0, "Training loss should be positive"

    # Run Validation
    print("Running validation on subset...")
    val_loss, val_auc = trainer_lib.validate(
        model, val_loader, criterion, config.DEVICE
    )
    print(f"Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")

    assert not np.isnan(val_loss), "Validation loss is NaN"
    assert 0 <= val_auc <= 1, "AUC must be between 0 and 1"

    # 6. Inference Demonstration
    print("\n--- Step 5: Inference Demonstration ---")
    model.eval()
    all_preds = []

    with torch.no_grad():
        for cat_inputs, cont_inputs in test_loader:
            cat_inputs = cat_inputs.to(config.DEVICE)
            cont_inputs = cont_inputs.to(config.DEVICE)

            outputs_list = model(cat_inputs, cont_inputs)

            # Ensemble prediction logic from trainer.py
            stream_probs = [torch.sigmoid(out) for out in outputs_list]
            avg_probs = torch.stack(stream_probs).mean(dim=0)

            all_preds.append(avg_probs.cpu().numpy())

    predictions = np.concatenate(all_preds).flatten()

    print(f"Generated {len(predictions)} predictions.")
    assert len(predictions) == len(
        test_subset
    ), "Number of predictions matches test subset size"
    assert np.all(
        (predictions >= 0) & (predictions <= 1)
    ), "Predictions must be probabilities [0, 1]"

    print("\n=== Demonstration Complete Successfully ===")


if __name__ == "__main__":
    run_demo()
