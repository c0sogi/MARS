import os
import sys
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import importlib

# --- Import Library Modules ---
import library.config as config
import library.utils as utils
import library.dataset as dataset
import library.model as model
import library.engine as engine


def main():
    print("=== Starting Demonstration Script ===")

    # ------------------------------------------------------------------------
    # 1. Configuration Patching for Speed
    # ------------------------------------------------------------------------
    print("\n[1] Patching configuration for fast execution...")

    # Patch config values
    config.EPOCHS = 1
    config.SEEDS = [42]
    config.DEBUG = True
    config.DEBUG_SAMPLE_SIZE = 64  # Small subset for demo
    config.BATCH_SIZE = 16  # Smaller batch size for demo

    # Since other modules import these variables using "from ... import ...",
    # we must patch them in the respective modules as well.
    dataset.DEBUG = config.DEBUG
    dataset.DEBUG_SAMPLE_SIZE = config.DEBUG_SAMPLE_SIZE
    dataset.BATCH_SIZE = config.BATCH_SIZE

    engine.EPOCHS = config.EPOCHS
    engine.SEEDS = config.SEEDS
    engine.BATCH_SIZE = config.BATCH_SIZE

    # Ensure working directory exists for debug cache
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    print(f"    Debug Mode: {dataset.DEBUG}")
    print(f"    Epochs: {engine.EPOCHS}")
    print(f"    Batch Size: {dataset.BATCH_SIZE}")

    # ------------------------------------------------------------------------
    # 2. Utility Verification
    # ------------------------------------------------------------------------
    print("\n[2] Verifying Utilities...")

    # Test Seed Everything
    utils.seed_everything(42)
    rnd_val = np.random.rand()
    print(f"    Random Seed Set. Random Value: {rnd_val:.4f}")

    # Test AUC Computation
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.1, 0.4, 0.35, 0.8])
    auc = utils.compute_auc(y_true, y_pred)
    print(f"    Computed AUC: {auc:.4f}")
    assert 0 <= auc <= 1.0, "AUC should be between 0 and 1"

    # ------------------------------------------------------------------------
    # 3. Data Loading Verification
    # ------------------------------------------------------------------------
    print("\n[3] Verifying Data Loading...")

    # Force reload of data to ensure debug sampling applies
    # We pass load_cached_data=False to force regeneration of debug cache
    train_loader, val_loader, test_loader = dataset.get_dataloaders(
        load_cached_data=False
    )

    print(f"    Train Batches: {len(train_loader)}")
    print(f"    Val Batches:   {len(val_loader)}")
    print(f"    Test Batches:  {len(test_loader)}")

    # Fetch one batch
    images, labels = next(iter(train_loader))

    print(f"    Image Batch Shape: {images.shape}")
    print(f"    Label Batch Shape: {labels.shape}")

    # Assertions
    assert images.shape == (config.BATCH_SIZE, 3, 32, 32), "Incorrect image batch shape"
    assert labels.shape == (config.BATCH_SIZE,), "Incorrect label batch shape"
    assert (
        images.max() <= 1.0 and images.min() >= 0.0
    ), "Images should be normalized to [0, 1]"
    assert labels.dtype == torch.float32, "Labels should be float32"

    # ------------------------------------------------------------------------
    # 4. Model Verification
    # ------------------------------------------------------------------------
    print("\n[4] Verifying Model Architecture...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Using device: {device}")

    net = model.NarrowSEResNet().to(device)

    # Count parameters
    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"    Model Parameters: {num_params}")

    # Forward pass check
    dummy_input = images.to(device)
    with torch.no_grad():
        output = net(dummy_input)

    print(f"    Output Shape: {output.shape}")
    assert output.shape == (config.BATCH_SIZE, 1), "Model output should be (B, 1)"
    assert not torch.isnan(output).any(), "Model output contains NaNs"

    # ------------------------------------------------------------------------
    # 5. Engine Verification (Train/Val Loop)
    # ------------------------------------------------------------------------
    print("\n[5] Verifying Training Engine...")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)

    # Run one epoch of training
    print("    Running train_one_epoch...")
    train_loss = engine.train_one_epoch(net, train_loader, criterion, optimizer, device)
    print(f"    Train Loss: {train_loss:.4f}")
    assert train_loss > 0, "Train loss should be positive"

    # Run validation
    print("    Running validate...")
    val_loss, val_auc = engine.validate(net, val_loader, criterion, device)
    print(f"    Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")

    # ------------------------------------------------------------------------
    # 6. Inference & Submission Verification
    # ------------------------------------------------------------------------
    print("\n[6] Verifying Inference and Submission...")

    # Simulate the generate_submission flow manually to use our patched config/objects

    # Train a model for one seed (Seed 42)
    seed = 42
    print(f"    Simulating full training for Seed {seed}...")
    best_model_path = engine.train_model(seed, train_loader, val_loader, device)
    assert os.path.exists(best_model_path), "Model file was not saved"

    # Load the best model
    net.load_state_dict(torch.load(best_model_path, map_location=device))
    net.to(device)
    net.eval()

    # Predict with TTA
    print("    Predicting with TTA...")
    preds = engine.predict_with_tta(net, test_loader, device)

    print(f"    Predictions shape: {preds.shape}")
    assert len(preds) == len(test_loader.dataset), "Prediction count mismatch"
    assert np.all(
        (preds >= 0) & (preds <= 1)
    ), "Predictions out of probability range [0, 1]"

    # Generate Submission CSV
    test_ids = test_loader.dataset.ids
    submission_df = pd.DataFrame({"id": test_ids, "has_cactus": preds})

    # Save to a demo path to avoid overwriting real submission if needed,
    # but here we use the config path
    submission_path = config.SUBMISSION_PATH
    submission_df.to_csv(submission_path, index=False)

    print(f"    Submission saved to: {submission_path}")

    # Verify file content
    saved_df = pd.read_csv(submission_path)
    print("    First 3 rows of submission:")
    print(saved_df.head(3))

    assert saved_df.shape == (len(test_loader.dataset), 2), "Submission shape mismatch"
    assert list(saved_df.columns) == ["id", "has_cactus"], "Submission columns mismatch"

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
