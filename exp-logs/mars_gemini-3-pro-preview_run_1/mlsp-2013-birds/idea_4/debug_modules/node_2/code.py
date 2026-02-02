import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# -------------------------------------------------------------------------
# 1. Configuration Override
# -------------------------------------------------------------------------
# We import the Config class and modify it *before* using other library modules
# to ensure the demo runs quickly and uses a temporary directory.
from library.config import Config

print("Configuring demo parameters...")
Config.DEBUG = True
Config.DEBUG_SUBSET_SIZE = 10  # Use only 10 samples for speed
Config.EPOCHS = 1  # Run only 1 epoch
Config.BATCH_SIZE = 2  # Small batch size
Config.NUM_WORKERS = 0  # Disable multiprocessing for simple demo
Config.PRETRAINED = False  # Do not download weights
Config.WORKING_DIR = "./working/demo_run"  # Separate working dir
Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

# Clean up demo directory if it exists
if os.path.exists(Config.WORKING_DIR):
    shutil.rmtree(Config.WORKING_DIR)
os.makedirs(Config.WORKING_DIR, exist_ok=True)

# -------------------------------------------------------------------------
# 2. Import Library Modules
# -------------------------------------------------------------------------
from library import utils
from library import dataset
from library import model as model_lib
from library import train as train_lib


def main():
    print("\n=== Library Usage Demonstration ===\n")

    # ---------------------------------------------------------------------
    # 3. Utilities Demonstration
    # ---------------------------------------------------------------------
    print("--- Testing Utilities ---")

    # Demonstrate seeding
    utils.seed_everything(Config.SEED)
    print("Random seed set.")

    # Demonstrate metric calculation
    # Create dummy multi-label data (3 samples, 3 classes)
    y_true_dummy = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
    y_pred_dummy = np.array([[0.9, 0.1, 0.8], [0.2, 0.8, 0.1], [0.8, 0.7, 0.2]])

    auc_score = utils.calculate_metric(y_true_dummy, y_pred_dummy)
    print(f"Calculated Dummy AUC: {auc_score:.4f}")

    # Validation
    assert isinstance(auc_score, float), "AUC score should be a float"
    assert 0.0 <= auc_score <= 1.0, "AUC score must be between 0 and 1"

    # ---------------------------------------------------------------------
    # 4. Dataset Demonstration
    # ---------------------------------------------------------------------
    print("\n--- Testing Dataset ---")

    # Load datasets (Debug mode is active, so this loads a subset)
    # We set load_cached_data=False to force processing logic execution
    print("Loading datasets (forcing processing)...")
    train_ds, val_ds, test_ds = dataset.get_datasets(load_cached_data=False)

    print(f"Train Dataset Size: {len(train_ds)}")
    print(f"Val Dataset Size: {len(val_ds)}")

    # Validation
    assert len(train_ds) == Config.DEBUG_SUBSET_SIZE, "Train dataset size mismatch"

    # Check item retrieval
    img, lbl = train_ds[0]
    print(f"Sample Image Shape: {img.shape}")
    print(f"Sample Label Shape: {lbl.shape}")

    # Expected shape: (Channels, Height, Width) -> (3, 256, 1280)
    assert img.shape == (3, Config.IMG_HEIGHT, Config.IMG_WIDTH), "Image shape mismatch"
    assert lbl.shape == (Config.NUM_CLASSES,), "Label shape mismatch"
    assert isinstance(img, torch.Tensor), "Image should be a Tensor"

    # ---------------------------------------------------------------------
    # 5. Model Demonstration
    # ---------------------------------------------------------------------
    print("\n--- Testing Model ---")

    # Instantiate the classifier
    # We use the backbone defined in Config (EfficientNet-B0) but without pretrained weights
    net = model_lib.BirdClassifier(backbone=Config.BACKBONE, pretrained=False)
    net.eval()  # Set to eval mode for deterministic output

    # Create a dummy batch
    dummy_batch = torch.randn(2, 3, Config.IMG_HEIGHT, Config.IMG_WIDTH)

    # Forward pass
    with torch.no_grad():
        logits = net(dummy_batch)

    print(f"Logits Shape: {logits.shape}")

    # Validation
    assert logits.shape == (2, Config.NUM_CLASSES), "Output logits shape mismatch"
    assert not torch.isnan(logits).any(), "Model output contains NaNs"

    # ---------------------------------------------------------------------
    # 6. Training Functions Demonstration
    # ---------------------------------------------------------------------
    print("\n--- Testing Training Functions ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    net.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(net.parameters(), lr=1e-3)

    # Create a DataLoader for the subset
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)

    # Test train_one_epoch
    print("Running single training epoch step...")
    loss = train_lib.train_one_epoch(
        net, train_loader, criterion, optimizer, device, alpha=0.0
    )
    print(f"Epoch Loss: {loss:.4f}")
    assert loss > 0, "Loss should be positive"

    # Test validate
    print("Running validation step...")
    val_loss, val_auc = train_lib.validate(net, train_loader, criterion, device)
    print(f"Validation Loss: {val_loss:.4f}, AUC: {val_auc:.4f}")

    # ---------------------------------------------------------------------
    # 7. Full Pipeline Demonstration
    # ---------------------------------------------------------------------
    print("\n--- Testing Full Training Pipeline ---")

    # The train() function encapsulates the entire workflow:
    # Data loading -> Model Init -> Training Loop -> Saving Best Model -> Inference
    # We expect it to finish quickly due to DEBUG=True and EPOCHS=1
    train_lib.train()

    # Verify artifacts
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    submission_path = Config.SUBMISSION_PATH

    print(f"Checking for model file: {best_model_path}")
    assert os.path.exists(best_model_path), "Best model file not found"

    print(f"Checking for submission file: {submission_path}")
    assert os.path.exists(submission_path), "Submission file not found"

    # Verify submission content
    df_sub = pd.read_csv(submission_path)
    print(f"Submission rows: {len(df_sub)}")
    assert (
        "Id" in df_sub.columns and "Probability" in df_sub.columns
    ), "Submission columns missing"

    # Check if Id format is correct (rec_id * 100 + species_id)
    # With DEBUG_SUBSET_SIZE=10, test set also has 10 samples.
    # 10 samples * 19 classes = 190 rows.
    expected_rows = Config.DEBUG_SUBSET_SIZE * Config.NUM_CLASSES
    assert (
        len(df_sub) == expected_rows
    ), f"Expected {expected_rows} rows in submission, got {len(df_sub)}"

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
