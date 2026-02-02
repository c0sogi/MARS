import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import shutil

# Import from the provided library
from library.config import Config
from library.dataset import BraTSDataset, get_dataloader
from library.model import VolumetricTransformer
from library.trainer import train_epoch, validate, set_seed


def run_demo():
    print("Initializing demonstration...")

    # ==========================================
    # 1. Configuration Override for Speed
    # ==========================================
    # We monkey-patch the Config class to run a fast, lightweight demo.
    print("Overriding configuration for fast execution...")
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 10  # Use only 10 subjects
    Config.IMG_SIZE = 128  # Reduce image size from 224
    Config.NUM_SLICES = 4  # Reduce sequence length from 24
    Config.BATCH_SIZE = 2  # Small batch size
    Config.NUM_EPOCHS = 1  # Only 1 epoch
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.WORKING_DIR = "./working/demo_run"
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pth")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seeds for reproducibility
    set_seed(Config.SEED)

    # ==========================================
    # 2. Dataset & DataLoader Verification
    # ==========================================
    print("\n--- Verifying Dataset & DataLoader ---")

    # Load metadata
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)

    # Create dataset (Debug mode limits to DEBUG_SAMPLE_SIZE)
    # We force load_cached_data=False to ensure the data preparation logic runs
    dataset = BraTSDataset(
        df_train.head(Config.DEBUG_SAMPLE_SIZE), split="train", load_cached_data=False
    )

    print(f"Dataset size: {len(dataset)}")

    # Verify Dataset Item Structure
    sample_img, sample_target = dataset[0]

    print(f"Sample Input Shape: {sample_img.shape}")
    print(f"Sample Target: {sample_target}")

    # Expected Shape: (Sequence, Channels, Height, Width)
    expected_shape = (
        Config.NUM_SLICES,
        Config.NUM_CHANNELS,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    )
    if sample_img.shape != expected_shape:
        raise AssertionError(
            f"Dataset output shape mismatch. Expected {expected_shape}, got {sample_img.shape}"
        )

    if not isinstance(sample_target, torch.Tensor):
        raise AssertionError("Target should be a torch.Tensor")

    # Verify DataLoader
    loader = get_dataloader(
        df_train,
        split="train",
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=False,
        debug=True,
    )

    batch_imgs, batch_targets = next(iter(loader))
    print(f"Batch Input Shape: {batch_imgs.shape}")
    print(f"Batch Target Shape: {batch_targets.shape}")

    # Expected Batch Shape: (Batch, Sequence, Channels, Height, Width)
    expected_batch_shape = (
        Config.BATCH_SIZE,
        Config.NUM_SLICES,
        Config.NUM_CHANNELS,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    )
    if batch_imgs.shape != expected_batch_shape:
        raise AssertionError(
            f"DataLoader batch shape mismatch. Expected {expected_batch_shape}, got {batch_imgs.shape}"
        )

    # ==========================================
    # 3. Model Verification
    # ==========================================
    print("\n--- Verifying Model Architecture ---")

    device = Config.DEVICE
    model = VolumetricTransformer().to(device)

    # Move batch to device
    batch_imgs = batch_imgs.to(device)

    # Forward pass
    logits = model(batch_imgs)

    print(f"Logits Shape: {logits.shape}")

    # Expected Output: (Batch, 1)
    if logits.shape != (Config.BATCH_SIZE, 1):
        raise AssertionError(
            f"Model output shape mismatch. Expected {(Config.BATCH_SIZE, 1)}, got {logits.shape}"
        )

    # ==========================================
    # 4. Training Loop Verification
    # ==========================================
    print("\n--- Verifying Training Step ---")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    # Run a single training epoch
    train_loss = train_epoch(model, loader, optimizer, criterion, device)

    print(f"Training Loss: {train_loss}")

    if not isinstance(train_loss, float) or np.isnan(train_loss):
        raise AssertionError("Training loss is invalid (NaN or not float).")

    # ==========================================
    # 5. Validation Loop Verification
    # ==========================================
    print("\n--- Verifying Validation Step ---")

    # Create validation loader
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)
    val_loader = get_dataloader(
        df_val,
        split="val",
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=False,
        debug=True,
    )

    val_loss, val_auc = validate(model, val_loader, criterion, device)

    print(f"Validation Loss: {val_loss}")
    print(f"Validation AUC: {val_auc}")

    if not (0.0 <= val_auc <= 1.0):
        raise AssertionError(f"Validation AUC {val_auc} is out of bounds [0, 1].")

    # ==========================================
    # 6. Inference / Submission Logic Check
    # ==========================================
    print("\n--- Verifying Inference Logic ---")

    # Save the model state
    torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)

    # Load Test Metadata
    df_test = pd.read_csv(Config.TEST_METADATA_PATH).head(Config.DEBUG_SAMPLE_SIZE)
    test_loader = get_dataloader(
        df_test,
        split="test",
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=False,
        debug=False,  # We manually subsetted df_test above
    )

    model.eval()
    predictions = []

    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            out = model(inputs)
            probs = torch.sigmoid(out)
            predictions.extend(probs.cpu().numpy().flatten())

    print(f"Number of predictions: {len(predictions)}")

    if len(predictions) != len(df_test):
        raise AssertionError(
            f"Prediction count mismatch. Expected {len(df_test)}, got {len(predictions)}"
        )

    # Create dummy submission
    sub_df = pd.DataFrame(
        {"BraTS21ID": df_test["BraTS21ID"], "MGMT_value": predictions}
    )
    output_csv = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    sub_df.to_csv(output_csv, index=False)

    print(f"Demo submission saved to {output_csv}")
    print("\nAll verification steps passed successfully.")


if __name__ == "__main__":
    run_demo()
