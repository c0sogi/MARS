import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.optim.swa_utils import AveragedModel
import pandas as pd

# Import provided library modules
from library import config, utils, data_loader, network, training_utils


def main():
    print("=== Iceberg Classifier Library Demo ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Override
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for demo...")

    # Redirect outputs to a demo directory to avoid cluttering real workspaces
    config.WORKING_DIR = "./working/demo_run"
    config.CACHE_DIR = os.path.join(config.WORKING_DIR, "cache")
    config.CHECKPOINT_DIR = os.path.join(config.WORKING_DIR, "checkpoints")
    config.SUBMISSION_DIR = os.path.join(config.WORKING_DIR, "submission")

    # Clean up previous demo runs
    if os.path.exists(config.WORKING_DIR):
        shutil.rmtree(config.WORKING_DIR)

    os.makedirs(config.CACHE_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)

    # Reduce parameters for speed
    config.BATCH_SIZE = 4
    config.MAX_EPOCHS_PHASE_1 = 1
    config.SWA_EPOCHS = 1
    config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Set reproducibility
    utils.seed_everything(seed=42)
    device = utils.get_device()
    print(f"    Device: {device}")
    print(f"    Output Directory: {config.WORKING_DIR}")

    # -------------------------------------------------------------------------
    # 2. Data Loading & Verification
    # -------------------------------------------------------------------------
    print("\n[2] Loading and Verifying Datasets...")

    # Load Train and Val datasets (force processing from scratch to test logic)
    # We use a small subset of indices for the actual loaders to speed up execution
    train_ds = data_loader.load_dataset("train", load_cached_data=False)
    val_ds = data_loader.load_dataset("val", load_cached_data=False)

    print(f"    Total Train Samples: {len(train_ds)}")
    print(f"    Total Val Samples: {len(val_ds)}")

    # Verify Data Item Structure
    img, ang, lbl = train_ds[0]

    # Check Image Shape: (Channels, Height, Width) -> (3, 224, 224) due to resizing
    assert img.shape == (
        3,
        config.IMAGE_SIZE,
        config.IMAGE_SIZE,
    ), f"Image shape mismatch. Expected (3, {config.IMAGE_SIZE}, {config.IMAGE_SIZE}), got {img.shape}"

    # Check Types
    assert isinstance(ang, torch.Tensor), "Angle must be a torch.Tensor"
    assert isinstance(lbl, torch.Tensor), "Label must be a torch.Tensor"

    print("    Data structure verification passed.")

    # Create small subsets for the demo loop
    subset_indices = list(range(12))  # Use 12 samples
    train_subset = Subset(train_ds, subset_indices)
    val_subset = Subset(val_ds, subset_indices)

    train_loader = DataLoader(train_subset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=config.BATCH_SIZE, shuffle=False)

    # -------------------------------------------------------------------------
    # 3. Model Initialization & Forward Pass
    # -------------------------------------------------------------------------
    print("\n[3] Initializing Model...")
    model = network.IcebergResNet18().to(device)

    # Verify Forward Pass with Dummy Data
    dummy_imgs = torch.randn(
        config.BATCH_SIZE, 3, config.IMAGE_SIZE, config.IMAGE_SIZE
    ).to(device)
    dummy_angs = torch.randn(config.BATCH_SIZE).to(device)

    with torch.no_grad():
        logits = model(dummy_imgs, dummy_angs)

    assert logits.shape == (
        config.BATCH_SIZE,
        1,
    ), f"Output shape mismatch. Expected ({config.BATCH_SIZE}, 1), got {logits.shape}"

    print("    Model initialized and forward pass verified.")

    # -------------------------------------------------------------------------
    # 4. Training Loop
    # -------------------------------------------------------------------------
    print("\n[4] Running Training Loop (1 Epoch)...")
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    loss, acc = training_utils.train_one_epoch(
        model, train_loader, optimizer, criterion, device, epoch=1
    )
    print(f"    Epoch 1 Finished. Loss: {loss:.4f}, Accuracy: {acc:.4f}")

    # Save Checkpoint
    ckpt_name = "demo_checkpoint.pth"
    training_utils.save_checkpoint(model, optimizer, 1, loss, ckpt_name)
    assert os.path.exists(
        os.path.join(config.CHECKPOINT_DIR, ckpt_name)
    ), "Checkpoint file not created."
    print(f"    Checkpoint saved to {ckpt_name}.")

    # -------------------------------------------------------------------------
    # 5. SWA (Stochastic Weight Averaging) Step
    # -------------------------------------------------------------------------
    print("\n[5] Testing SWA Update...")
    swa_model = AveragedModel(model)
    training_utils.swa_step(model, swa_model)

    # Update BN statistics
    training_utils.update_swa_batch_norm(swa_model, train_loader, device)
    print("    SWA parameters and BN statistics updated.")

    # -------------------------------------------------------------------------
    # 6. Evaluation (with TTA)
    # -------------------------------------------------------------------------
    print("\n[6] Evaluating Model (with TTA)...")
    val_loss, val_acc, val_probs, val_targets = training_utils.evaluate(
        model, val_loader, criterion, device, use_tta=True
    )
    print(f"    Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

    assert len(val_probs) == len(
        val_subset
    ), "Number of predictions does not match subset size."

    # -------------------------------------------------------------------------
    # 7. Prediction & Submission
    # -------------------------------------------------------------------------
    print("\n[7] Generating Test Predictions...")
    test_ds = data_loader.load_dataset("test", load_cached_data=False)

    # Subset test data
    test_subset = Subset(test_ds, subset_indices)
    test_loader = DataLoader(test_subset, batch_size=config.BATCH_SIZE, shuffle=False)

    # Predict
    preds, ids = training_utils.predict(model, test_loader, device, use_tta=False)

    assert len(preds) == len(ids), "Mismatch between predictions and IDs."

    # Write Submission
    sub_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
    training_utils.write_submission(ids, preds, sub_path)

    # Verify CSV
    df = pd.read_csv(sub_path)
    assert list(df.columns) == ["id", "is_iceberg"], "Submission columns incorrect."
    assert len(df) == len(subset_indices), "Submission row count incorrect."

    print(f"    Submission generated at {sub_path}")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
