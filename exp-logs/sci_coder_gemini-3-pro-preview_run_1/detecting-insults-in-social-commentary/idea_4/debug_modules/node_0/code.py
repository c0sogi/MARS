import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import provided library components
from library.config import Config
from library.utils import seed_everything
from library.data_processing import get_datasets
from library.model import HybridDeberta
from library.trainer import Trainer


def main():
    print("=== Starting Demonstration Script ===")

    # ---------------------------------------------------------
    # 1. Configuration Overrides for Fast Execution
    # ---------------------------------------------------------
    print("[1/7] Configuring environment for fast demo run...")

    # Enable debug mode to use a small subset of data (100 train, 50 val, 50 test)
    Config.debug = True

    # Reduce training duration
    Config.epochs = 1

    # Reduce batch size and sequence length for speed and memory efficiency
    Config.batch_size = 4
    Config.max_len = 32

    # Reduce SVD components because debug dataset size (100) < default components (256)
    # TruncatedSVD requires n_components < n_samples
    Config.svd_components = 10

    # Enable AWP immediately to demonstrate it works (default starts at epoch 2)
    Config.awp_start_epoch = 0

    # Ensure reproducibility
    seed_everything(Config.seed)

    # Clean up any existing cache files in working directory to ensure
    # we regenerate features with the new 'svd_components' size.
    cache_files = ["train_svd.npy", "val_svd.npy", "test_svd.npy"]
    for f in cache_files:
        path = os.path.join(Config.working_dir, f)
        if os.path.exists(path):
            os.remove(path)

    print("      Configuration updated: Debug=True, Epochs=1, Batch=4, SVD=10")

    # ---------------------------------------------------------
    # 2. Data Processing
    # ---------------------------------------------------------
    print("[2/7] Loading and processing data...")

    # load_cached_data=False ensures we run the FeatureEngineer logic
    train_dataset, val_dataset, test_dataset, tokenizer = get_datasets(
        load_cached_data=False, debug=Config.debug
    )

    # Validation: Check Dataset Structure
    print("      Verifying dataset integrity...")
    assert len(train_dataset) > 0, "Training dataset is empty."

    sample = train_dataset[0]
    required_keys = ["input_ids", "attention_mask", "svd_feat", "label"]
    for key in required_keys:
        assert key in sample, f"Missing key in dataset sample: {key}"

    # Check shapes
    assert (
        sample["input_ids"].shape[0] == Config.max_len
    ), f"Input IDs shape mismatch. Expected {Config.max_len}, got {sample['input_ids'].shape[0]}"
    assert (
        sample["svd_feat"].shape[0] == Config.svd_components
    ), f"SVD feature shape mismatch. Expected {Config.svd_components}, got {sample['svd_feat'].shape[0]}"

    print("      Dataset verification passed.")

    # ---------------------------------------------------------
    # 3. DataLoader Setup
    # ---------------------------------------------------------
    print("[3/7] Setting up DataLoaders...")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=0
    )

    # ---------------------------------------------------------
    # 4. Model Initialization & Verification
    # ---------------------------------------------------------
    print("[4/7] Initializing HybridDeberta model...")

    device = Config.device
    # Initialize model (pretrained=True downloads weights, but we assume environment capability)
    model = HybridDeberta(pretrained=True)
    model.to(device)

    # Validation: Test Forward Pass
    print("      Verifying model forward pass...")
    batch = next(iter(train_loader))
    input_ids = batch["input_ids"].to(device)
    mask = batch["attention_mask"].to(device)
    svd = batch["svd_feat"].to(device)

    with torch.no_grad():
        output = model(input_ids, mask, svd)

    assert output.shape == (
        Config.batch_size,
    ), f"Model output shape mismatch. Expected {(Config.batch_size,)}, got {output.shape}"
    print("      Model forward pass successful.")

    # ---------------------------------------------------------
    # 5. Training Loop
    # ---------------------------------------------------------
    print("[5/7] Starting training loop (Trainer)...")

    trainer = Trainer(train_loader, val_loader, device=device)

    # Run training
    # This handles AWP, Validation, and Model Saving internally
    best_model_path, best_auc = trainer.train()

    # Validation: Check Training Results
    assert os.path.exists(
        best_model_path
    ), f"Best model file not found at {best_model_path}"
    assert isinstance(best_auc, float), "Best AUC is not a float."
    print(f"      Training complete. Best AUC: {best_auc:.4f}")

    # ---------------------------------------------------------
    # 6. Inference
    # ---------------------------------------------------------
    print("[6/7] Running inference on test set...")

    predictions = trainer.predict(test_loader, best_model_path)

    # Validation: Check Predictions
    assert len(predictions) == len(
        test_dataset
    ), f"Prediction count ({len(predictions)}) matches test set size ({len(test_dataset)})"
    assert np.all(
        (predictions >= 0) & (predictions <= 1)
    ), "Predictions contain values outside [0, 1] range."

    print("      Inference successful.")

    # ---------------------------------------------------------
    # 7. Submission Generation
    # ---------------------------------------------------------
    print("[7/7] Generating submission file...")

    # Load test metadata to align predictions
    # Note: Since we used debug=True, we must slice the original test CSV to match the subset
    df_test_full = pd.read_csv(Config.test_path)
    if Config.debug:
        df_test = df_test_full.head(len(predictions)).copy()
    else:
        df_test = df_test_full.copy()

    df_test["Insult"] = predictions

    # Format: Insult, Date, Comment
    submission_df = df_test[["Insult", "Date", "Comment"]]

    # Save
    os.makedirs(Config.submission_dir, exist_ok=True)
    submission_df.to_csv(Config.submission_file, index=False)

    assert os.path.exists(Config.submission_file), "Submission file was not created."
    print(f"      Submission saved to {Config.submission_file}")

    print("=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
