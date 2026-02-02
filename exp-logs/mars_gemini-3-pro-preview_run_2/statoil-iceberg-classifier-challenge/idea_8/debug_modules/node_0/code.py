import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import shutil

# Import library components
from library.config import Config
from library.utils import seed_everything, save_checkpoint, load_checkpoint
from library.data_loader import get_loader, IcebergDataset, process_and_cache_data
from library.model import MSAHN
from library.train_eval import train_one_epoch, validate

if __name__ == "__main__":
    # 1. Setup and Configuration
    print("--- 1. Setup and Configuration ---")
    seed_everything(42)

    # Override Config for rapid demonstration
    Config.DEBUG = True
    Config.SUBSET_SIZE = 32  # Small subset for speed
    Config.BATCH_SIZE = 8
    Config.EPOCHS = 1
    Config.WORKING_DIR = "./working/demo_execution"
    Config.PROCESSED_DATA_FILE = os.path.join(Config.WORKING_DIR, "processed_data.npz")
    Config.MODEL_CHECKPOINT_PREFIX = os.path.join(Config.WORKING_DIR, "demo_model")
    Config.SUBMISSION_FILE = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    # Ensure directories exist
    Config.setup()
    print(f"Working directory set to: {Config.WORKING_DIR}")

    # 2. Data Loading Demonstration
    print("\n--- 2. Data Loading Demonstration ---")

    # Load metadata
    train_meta_path = Config.TRAIN_META_FILE
    if not os.path.exists(train_meta_path):
        raise FileNotFoundError(f"Metadata file not found: {train_meta_path}")

    df_train = pd.read_csv(train_meta_path)
    print(f"Loaded train metadata with {len(df_train)} rows.")

    # Initialize DataLoader
    # This will trigger process_and_cache_data which reads the JSONs and creates the NPZ
    print("Initializing DataLoader (this may take a moment to process raw JSONs)...")
    train_loader = get_loader(
        df_train,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        augment=True,
        load_cached_data=True,
    )

    # Fetch one batch to verify
    images, inc_angles, labels = next(iter(train_loader))

    print(
        f"Batch shapes - Images: {images.shape}, Angles: {inc_angles.shape}, Labels: {labels.shape}"
    )

    # Assertions for Data Loader
    assert images.shape == (Config.BATCH_SIZE, 3, 75, 75), "Incorrect image batch shape"
    assert inc_angles.shape == (
        Config.BATCH_SIZE,
        1,
    ), "Incorrect incidence angle batch shape"
    assert labels.shape == (Config.BATCH_SIZE, 1), "Incorrect label batch shape"
    assert images.dtype == torch.float32, "Images should be float32"
    print("Data Loader verification passed.")

    # 3. Model Instantiation and Forward Pass
    print("\n--- 3. Model Instantiation and Forward Pass ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = MSAHN().to(device)

    # Move batch to device
    images = images.to(device)
    inc_angles = inc_angles.to(device)
    labels = labels.to(device)

    # Forward pass
    outputs = model(images, inc_angles)

    print(f"Model output shape: {outputs.shape}")

    # Assertions for Model
    assert outputs.shape == (Config.BATCH_SIZE, 1), "Model output shape mismatch"
    assert not torch.isnan(outputs).any(), "Model produced NaN outputs"
    print("Model forward pass verification passed.")

    # 4. Training Loop Demonstration
    print("\n--- 4. Training Loop Demonstration ---")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Run one training epoch
    print("Running training step...")
    train_loss, train_acc = train_one_epoch(
        train_loader, model, criterion, optimizer, device, epoch=0
    )
    print(f"Train Result - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")

    # Assertions for Training
    assert train_loss > 0, "Training loss should be positive"
    assert 0 <= train_acc <= 1, "Training accuracy should be between 0 and 1"

    # Run validation step (using the same loader for demo purposes to save time loading val set)
    print("Running validation step...")
    val_loss, val_acc = validate(train_loader, model, criterion, device)
    print(f"Val Result - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

    print("Training loop verification passed.")

    # 5. Checkpoint Save/Load Demonstration
    print("\n--- 5. Checkpoint Save/Load Demonstration ---")

    ckpt_path = f"{Config.MODEL_CHECKPOINT_PREFIX}_test.pth"

    # Save
    save_state = {
        "epoch": 1,
        "state_dict": model.state_dict(),
        "best_loss": val_loss,
        "optimizer": optimizer.state_dict(),
    }
    save_checkpoint(save_state, is_best=True, filename=ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")

    # Load
    new_model = MSAHN().to(device)
    start_epoch, best_loss = load_checkpoint(ckpt_path, new_model)

    # Verify loaded weights match
    original_param = next(model.parameters())
    loaded_param = next(new_model.parameters())

    assert torch.equal(
        original_param, loaded_param
    ), "Model weights did not match after loading"
    assert start_epoch == 1, "Incorrect start epoch loaded"
    assert best_loss == val_loss, "Incorrect best loss loaded"

    print("Checkpoint verification passed.")

    # 6. Inference Demonstration
    print("\n--- 6. Inference Demonstration ---")

    # Switch to eval mode
    new_model.eval()

    # Create dummy test loader (using subset of train for demo)
    test_loader = get_loader(
        df_train.head(Config.BATCH_SIZE),
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        augment=False,
        load_cached_data=True,
    )

    predictions = []

    with torch.no_grad():
        for batch_imgs, batch_inc, batch_lbls in test_loader:
            batch_imgs = batch_imgs.to(device)
            batch_inc = batch_inc.to(device)

            logits = new_model(batch_imgs, batch_inc)
            probs = torch.sigmoid(logits).cpu().numpy()
            predictions.extend(probs.flatten())

    print(f"Generated {len(predictions)} predictions.")
    assert len(predictions) > 0, "No predictions generated"
    assert 0 <= predictions[0] <= 1, "Probabilities must be between 0 and 1"

    print("Inference verification passed.")

    print("\n=== All Demonstrations Completed Successfully ===")
