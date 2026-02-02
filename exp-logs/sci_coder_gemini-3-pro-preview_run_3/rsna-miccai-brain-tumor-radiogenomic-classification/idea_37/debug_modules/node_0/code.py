import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

# Import provided library modules
from library import config, data_loader, model, utils


def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    print("Initializing Demo Script...")

    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Override config parameters for a fast demonstration
    set_seed(config.SEED)

    # Create a separate cache directory for this demo to avoid conflicts
    demo_cache_dir = os.path.join(config.WORKING_DIR, "demo_cache")
    if os.path.exists(demo_cache_dir):
        shutil.rmtree(demo_cache_dir)
    os.makedirs(demo_cache_dir, exist_ok=True)

    # Monkey-patch the config module
    config.CACHE_DIR = demo_cache_dir
    config.DEBUG = True
    config.DEBUG_SAMPLE_SIZE = 20  # Process only 20 samples per split
    config.BATCH_SIZE = 4
    config.N_EPOCHS = 1
    config.NUM_WORKERS = 2

    print(f"Configured for demo: Debug={config.DEBUG}, BatchSize={config.BATCH_SIZE}")
    print(f"Cache Directory: {config.CACHE_DIR}")

    # ==========================================
    # 2. Data Loading & Verification
    # ==========================================
    print("\n[Step 2] Loading Data...")

    # Load Train and Validation DataLoaders
    # This triggers _process_data which loads DICOMs, normalizes, and caches npy files
    train_loader = data_loader.get_dataloader("train", shuffle=True)
    val_loader = data_loader.get_dataloader("val", shuffle=False)

    # Verify Train Loader
    try:
        # Fetch one batch
        x_even, x_odd, y = next(iter(train_loader))

        # Check Shapes
        # Expected: (Batch, 64, 224, 224)
        assert x_even.shape == (
            config.BATCH_SIZE,
            config.IN_CHANS,
            config.IMG_SIZE,
            config.IMG_SIZE,
        ), f"Incorrect X_even shape: {x_even.shape}"
        assert x_odd.shape == (
            config.BATCH_SIZE,
            config.IN_CHANS,
            config.IMG_SIZE,
            config.IMG_SIZE,
        ), f"Incorrect X_odd shape: {x_odd.shape}"
        assert y.shape == (config.BATCH_SIZE,), f"Incorrect y shape: {y.shape}"

        print(f"Data Validation Passed. Batch Shape: {x_even.shape}")

    except StopIteration:
        raise RuntimeError(
            "Train loader is empty! Check metadata or debug sample size."
        )

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    print("\n[Step 3] Initializing Siamese SNR-Net...")

    device = config.DEVICE
    net = model.SiameseSNRNet().to(device)

    # Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=config.LEARNING_RATE)

    print(f"Model created on {device}. Backbone: {config.BACKBONE}")

    # ==========================================
    # 4. Training Loop (Demo)
    # ==========================================
    print("\n[Step 4] Starting Training (1 Epoch)...")

    net.train()
    train_loss = 0.0
    train_steps = 0

    for batch_idx, (x_even, x_odd, targets) in enumerate(train_loader):
        x_even, x_odd = x_even.to(device), x_odd.to(device)
        targets = targets.to(device).unsqueeze(1)  # (B, 1)

        optimizer.zero_grad()

        # Forward Pass
        logits = net(x_even, x_odd)

        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_steps += 1

    avg_train_loss = train_loss / train_steps if train_steps > 0 else 0
    print(f"Epoch 1 Completed. Avg Loss: {avg_train_loss:.4f}")

    # ==========================================
    # 5. Validation & Evaluation
    # ==========================================
    print("\n[Step 5] Validating...")

    net.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_even, x_odd, targets in val_loader:
            x_even, x_odd = x_even.to(device), x_odd.to(device)

            logits = net(x_even, x_odd)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_preds.extend(probs)
            all_targets.extend(targets.numpy())

    all_preds = np.array(all_preds).flatten()
    all_targets = np.array(all_targets).flatten()

    # Calculate Metric
    # Handle edge case in debug mode where only 1 class might be sampled
    if len(np.unique(all_targets)) > 1:
        auc_score = roc_auc_score(all_targets, all_preds)
        print(f"Validation AUC: {auc_score:.4f}")
    else:
        print("Validation AUC: N/A (Only one class present in debug subset)")

    # ==========================================
    # 6. Test Inference & Submission
    # ==========================================
    print("\n[Step 6] Generating Test Predictions...")

    # Load Test Loader (No labels returned)
    test_loader = data_loader.get_dataloader("test", shuffle=False)

    # We need to retrieve IDs to map predictions back to BraTS21ID
    # The dataloader doesn't return IDs in __getitem__, so we load them from cache
    # or we can modify the dataset. For this demo, we'll load the IDs from the cache file
    # that was generated during get_dataloader('test').
    cache_ids_path = os.path.join(config.CACHE_DIR, "ids_test.npy")
    if not os.path.exists(cache_ids_path):
        raise FileNotFoundError("Test IDs cache not found.")

    test_ids = np.load(cache_ids_path)

    test_preds = []
    net.eval()

    with torch.no_grad():
        for x_even, x_odd in test_loader:
            x_even, x_odd = x_even.to(device), x_odd.to(device)
            logits = net(x_even, x_odd)
            probs = torch.sigmoid(logits).cpu().numpy()
            test_preds.extend(probs)

    test_preds = np.array(test_preds).flatten()

    # Ensure lengths match (might differ slightly if drop_last or similar, but here exact)
    # In debug mode, test_loader processes DEBUG_SAMPLE_SIZE items.
    # test_ids also comes from _process_data which respects DEBUG_SAMPLE_SIZE.
    assert len(test_ids) == len(
        test_preds
    ), f"Mismatch: IDs={len(test_ids)}, Preds={len(test_preds)}"

    # Create Submission DataFrame
    submission_df = pd.DataFrame({"BraTS21ID": test_ids, "MGMT_value": test_preds})

    # Save
    demo_sub_path = os.path.join(config.WORKING_DIR, "demo_submission.csv")
    submission_df.to_csv(demo_sub_path, index=False)

    print(f"Submission saved to: {demo_sub_path}")
    print(submission_df.head())

    print("\nDemo Completed Successfully.")


if __name__ == "__main__":
    main()
