import os
import sys
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.utils import set_seed, save_checkpoint, load_checkpoint
from library.dataset import preprocess_data, get_fold_datasets, get_test_dataset
from library.model import AGICNN
from library.train import train_one_epoch, validate


def run_demo():
    print("=== Starting Demo Execution ===")

    # 1. Configuration Overrides for Demo
    # We modify Config attributes to isolate this run and make it fast
    print("Configuring demo parameters...")
    Config.WORKING_DIR = "./working"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "demo_cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.CACHE_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.CACHE_DIR, "submission")

    # Speed optimizations
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.DEBUG = True

    # Ensure directories exist
    Config.setup()

    # Set seed for reproducibility
    set_seed(Config.SEED)

    # 2. Data Processing
    print("\n--- Testing Data Processing ---")
    # Force reload to demonstrate processing logic, but save to demo cache
    # Note: preprocess_data reads the full JSONs from input.
    # Since input is read-only and small (~1600 records), this is acceptable.
    X_train, y_train, angles_train, ids_train, X_test, angles_test, ids_test = (
        preprocess_data(load_cached_data=False)
    )

    # Verify Data Shapes
    assert len(X_train) == len(y_train) == len(angles_train)
    assert X_train.shape[1:] == (
        3,
        75,
        75,
    ), f"Expected (3, 75, 75), got {X_train.shape[1:]}"
    print(f"Data loaded. Train size: {len(X_train)}, Test size: {len(X_test)}")

    # SUBSETTING for Speed
    # We will use only 40 samples for the rest of the demo
    subset_size = 40
    print(f"Subsetting data to {subset_size} samples for rapid testing...")
    X_train = X_train[:subset_size]
    y_train = y_train[:subset_size]
    angles_train = angles_train[:subset_size]
    ids_train = ids_train[:subset_size]

    # Introduce a NaN in angles to test imputation if not already present
    angles_train[0] = np.nan

    # 3. Dataset & DataLoader
    print("\n--- Testing Dataset & Imputation ---")
    # Get Fold 0
    train_ds, val_ds = get_fold_datasets(
        X_train, y_train, angles_train, ids_train, fold=0, num_folds=5, seed=Config.SEED
    )

    print(f"Fold 0 Split - Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Check item retrieval
    img, angle, label = train_ds[0]

    # Assertions
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 75, 75)
    assert not torch.isnan(angle), "Angle imputation failed: NaN found in dataset item"
    assert isinstance(label, torch.Tensor)

    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False)

    # 4. Model Instantiation
    print("\n--- Testing Model Initialization & Forward Pass ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = AGICNN().to(device)

    # Create dummy batch
    dummy_imgs = img.unsqueeze(0).to(device)  # (1, 3, 75, 75)
    dummy_angle = angle.unsqueeze(0).to(device)  # (1,)

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_imgs, dummy_angle)

    # Assertions
    assert output.shape == (1,), f"Expected output shape (1,), got {output.shape}"
    print("Model forward pass successful.")

    # 5. Training Loop
    print("\n--- Testing Training Loop (1 Epoch) ---")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Train
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )
    print(f"Train Result -> Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
    assert not np.isnan(train_loss), "Training loss is NaN"

    # Validate
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    print(f"Val Result   -> Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
    assert not np.isnan(val_loss), "Validation loss is NaN"

    # 6. Checkpointing
    print("\n--- Testing Checkpointing ---")
    # Save
    state = {
        "epoch": 1,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_val_loss": val_loss,
        "fold": 0,
    }
    save_checkpoint(state, is_best=True, fold=0, checkpoint_dir=Config.CHECKPOINT_DIR)

    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "checkpoint_fold_0.pth")
    assert os.path.exists(checkpoint_path), "Checkpoint file not created"

    # Load
    model_new = AGICNN().to(device)
    loaded_state = load_checkpoint(
        model_new, fold=0, load_best=False, checkpoint_dir=Config.CHECKPOINT_DIR
    )

    # Verify weights match
    original_w = model.classifier.weight.data
    loaded_w = model_new.classifier.weight.data
    assert torch.equal(original_w, loaded_w), "Model weights mismatch after loading"
    print("Checkpoint save/load verified.")

    # 7. Test Inference
    print("\n--- Testing Inference Pipeline ---")
    # Subset test data
    X_test_sub = X_test[:10]
    angles_test_sub = angles_test[:10]
    ids_test_sub = ids_test[:10]

    # Create test dataset (uses global train stats for imputation)
    test_ds = get_test_dataset(X_test_sub, angles_test_sub, ids_test_sub, angles_train)
    test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False)

    model.eval()
    predictions = []
    with torch.no_grad():
        for images, angles, ids in test_loader:
            images = images.to(device)
            angles = angles.to(device)
            outputs = model(images, angles)
            probs = torch.sigmoid(outputs)
            predictions.extend(probs.cpu().numpy())

    assert len(predictions) == 10
    assert all(
        0.0 <= p <= 1.0 for p in predictions
    ), "Probabilities out of range [0, 1]"
    print("Inference successful.")

    # 8. Cleanup
    print("\n--- Cleanup ---")
    if os.path.exists(Config.CACHE_DIR):
        print(f"Removing demo cache: {Config.CACHE_DIR}")
        shutil.rmtree(Config.CACHE_DIR)

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
