import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# Ensure the current directory is in the path to import library modules
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, compute_pos_weight
from library.data import get_data_with_folds, get_loaders, get_test_loader
from library.models import get_model
from library.engine import train_one_epoch, validate, inference, save_predictions


def main():
    print("Initializing Demonstration Script...")

    # --- 1. Configuration Overrides for Speed ---
    # We modify the Config class attributes directly to run a fast demo
    print("Configuring environment for rapid demonstration...")
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 64  # Small subset for quick execution
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data
    Config.N_FOLDS = 2  # Minimal folds for demo
    Config.EPOCHS = 2  # Run only 2 epochs

    # Setup directories and seeds
    Config.setup()
    seed_everything(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")

    # --- 2. Data Pipeline ---
    print("\n--- Step 2: Data Pipeline ---")

    # Load data and assign folds
    # This uses the metadata files in ./metadata and caches splits to ./working
    df_folds = get_data_with_folds(load_cached_data=False)
    print(f"Total samples in dataset (with folds): {len(df_folds)}")

    # Get DataLoaders for Fold 0
    fold_idx = 0
    train_loader, val_loader = get_loaders(
        fold_idx,
        df_folds,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
    )

    print(f"Train Loader batches: {len(train_loader)}")
    print(f"Val Loader batches: {len(val_loader)}")

    # VALIDATION: Check batch structure
    dummy_inputs, dummy_targets = next(iter(train_loader))
    print(f"Batch Input Shape: {dummy_inputs.shape}")  # Expected: [8, 3, 224, 448]
    print(f"Batch Target Shape: {dummy_targets.shape}")  # Expected: [8, 19]

    assert dummy_inputs.shape == (
        Config.BATCH_SIZE,
        Config.CHANNELS,
        Config.IMAGE_SIZE[0],
        Config.IMAGE_SIZE[1],
    ), "Incorrect input shape from DataLoader"
    assert dummy_targets.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), "Incorrect target shape from DataLoader"

    # --- 3. Model Initialization ---
    print("\n--- Step 3: Model Initialization ---")

    # Instantiate a ResNet18 model
    model_name = "resnet18"
    model = get_model(model_name, pretrained=True)
    model.to(device)

    # VALIDATION: Check forward pass
    with torch.no_grad():
        dummy_out = model(dummy_inputs.to(device))

    print(f"Model Output Shape: {dummy_out.shape}")
    assert dummy_out.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), "Model output shape mismatch"

    # --- 4. Training Loop ---
    print("\n--- Step 4: Training Loop ---")

    # Compute class weights for loss function
    # In a real scenario, we'd use the full training set labels
    # Here we use the subset labels available in the loader's dataset
    y_train_subset = train_loader.dataset.labels
    pos_weight = compute_pos_weight(y_train_subset).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Run for a few epochs
    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{Config.EPOCHS}")
        train_loss = train_one_epoch(model, optimizer, train_loader, device, criterion)
        val_loss, val_auc = validate(model, val_loader, device, criterion)

        # VALIDATION: Ensure metrics are valid numbers
        assert not np.isnan(train_loss), "Training loss is NaN"
        assert not np.isnan(val_loss), "Validation loss is NaN"
        # AUC might be 0.0 if the subset lacks positive samples for some classes, which is handled by utils.py

    # --- 5. Inference & Submission ---
    print("\n--- Step 5: Inference & Submission ---")

    test_loader = get_test_loader(
        batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
    )
    print(f"Test Loader batches: {len(test_loader)}")

    # Run inference
    rec_ids, probs = inference(model, test_loader, device)

    print(f"Inference complete. Predictions shape: {probs.shape}")

    # VALIDATION: Check inference output
    assert len(rec_ids) == len(probs), "Mismatch between recording IDs and predictions"
    assert (
        probs.shape[1] == Config.NUM_CLASSES
    ), "Incorrect number of probability columns"

    # Save submission
    submission_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    save_predictions(rec_ids, probs, submission_path)

    print(f"Submission saved to: {submission_path}")

    # VALIDATION: Verify file creation and format
    assert os.path.exists(submission_path), "Submission file was not created"

    df_sub = pd.read_csv(submission_path)
    print(f"Submission rows: {len(df_sub)}")
    print(df_sub.head())

    expected_cols = ["Id", "Probability"]
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}"

    # Check Id format (should be roughly rec_id * 100 + species_id)
    # Just checking the first one
    first_id = df_sub.iloc[0]["Id"]
    assert isinstance(first_id, (int, np.integer)), "Id column should be integers"

    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    main()
