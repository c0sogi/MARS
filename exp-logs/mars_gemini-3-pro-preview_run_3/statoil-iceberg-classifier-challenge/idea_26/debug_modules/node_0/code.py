import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Import library components
from library.config import Config
from library.utils import seed_everything, save_checkpoint, load_checkpoint
from library.dataset import get_loaders
from library.model import MADResNet
from library.engine import train_one_epoch, evaluate, predict


def run_demo():
    print("=== Starting Library Demo ===")

    # 1. Configure for fast execution (Demo Mode)
    print("\n[1] Configuring environment...")
    Config.DEBUG = True
    Config.BATCH_SIZE = 4
    Config.NUM_EPOCHS = 1
    Config.IDEA_DIR = "./working/demo_run"
    Config.setup()  # Ensure the new directory exists

    # Set device manually if needed, though Config does it automatically
    device = torch.device(Config.DEVICE)
    print(f"    Device: {device}")
    print(f"    Working Directory: {Config.IDEA_DIR}")

    # 2. Reproducibility
    print("\n[2] Setting random seeds...")
    seed_everything(42)

    # 3. Data Loading
    print("\n[3] Initializing DataLoaders (Debug Mode)...")
    # This will trigger process_data(), creating cache files in Config.IDEA_DIR
    train_loader, val_loader, test_loader = get_loaders(
        batch_size=Config.BATCH_SIZE, debug=True
    )

    # Verify Data Shapes
    print("    Verifying training batch structure...")
    images, angles, labels = next(iter(train_loader))

    print(f"    Image Batch Shape: {images.shape}")  # Should be (4, 3, 75, 75)
    print(f"    Angle Batch Shape: {angles.shape}")  # Should be (4,)
    print(f"    Label Batch Shape: {labels.shape}")  # Should be (4,)

    assert images.shape == (Config.BATCH_SIZE, 3, 75, 75), "Incorrect image batch shape"
    assert angles.shape == (Config.BATCH_SIZE,), "Incorrect angle batch shape"
    assert labels.shape == (Config.BATCH_SIZE,), "Incorrect label batch shape"

    # 4. Model Initialization
    print("\n[4] Instantiating MADResNet model...")
    model = MADResNet().to(device)

    # Verify Forward Pass
    print("    Running dummy forward pass...")
    images = images.to(device)
    angles = angles.to(device)

    with torch.no_grad():
        outputs = model(images, angles)

    print(f"    Output Shape: {outputs.shape}")
    assert outputs.shape == (Config.BATCH_SIZE, 1), "Model output shape mismatch"

    # 5. Training Loop (Engine)
    print("\n[5] Testing Training Loop (Engine)...")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    loss, acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    print(f"    Train Epoch Result -> Loss: {loss:.4f}, Accuracy: {acc:.4f}")

    assert not np.isnan(loss), "Training loss is NaN"
    assert 0.0 <= acc <= 1.0, "Training accuracy out of bounds"

    # 6. Evaluation (Engine)
    print("\n[6] Testing Evaluation Loop (Engine)...")
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    print(f"    Val Epoch Result   -> Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

    assert not np.isnan(val_loss), "Validation loss is NaN"

    # 7. Checkpointing
    print("\n[7] Testing Checkpointing...")
    fold_idx = 0

    # Save
    save_checkpoint(
        state={"state_dict": model.state_dict(), "epoch": 1},
        is_best=True,
        fold=fold_idx,
        output_dir=Config.IDEA_DIR,
    )

    checkpoint_path = os.path.join(Config.IDEA_DIR, f"model_best_fold_{fold_idx}.pth")
    assert os.path.exists(checkpoint_path), "Checkpoint file was not created"

    # Load
    print("    Loading checkpoint back...")
    loaded_checkpoint = load_checkpoint(checkpoint_path, model, device=device)
    assert "state_dict" in loaded_checkpoint, "Loaded checkpoint missing state_dict"
    print("    Checkpoint loaded successfully.")

    # 8. Inference and Submission
    print("\n[8] Testing Inference and Submission generation...")
    preds = predict(model, test_loader, device)

    print(f"    Predictions shape: {preds.shape}")
    # In debug mode, test_loader size is Config.DEBUG_SUBSET_SIZE (100)
    expected_len = Config.DEBUG_SUBSET_SIZE
    # However, DataLoader might drop last or not depending on batch size,
    # but test loader usually doesn't drop last.
    # Let's verify we have predictions for all items in the loader.
    assert len(preds) == len(
        test_loader.dataset
    ), f"Prediction count {len(preds)} mismatch with dataset size {len(test_loader.dataset)}"

    # Create submission DataFrame
    # We need the IDs from the dataset
    test_ids = test_loader.dataset.ids

    submission_df = pd.DataFrame({"id": test_ids, "is_iceberg": preds})

    print("    Sample Submission Head:")
    print(submission_df.head())

    # Save to demo directory
    sub_path = os.path.join(Config.IDEA_DIR, "demo_submission.csv")
    submission_df.to_csv(sub_path, index=False)
    print(f"    Submission saved to {sub_path}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
