import sys
import os
import io
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

# 1. Configuration Override
# We import config first to modify parameters before other modules use them.
from library import config

# Set random seeds for reproducibility
from library import utils

utils.seed_everything(42)

# Override global config variables to ensure the demo runs quickly.
# The train.py script runs on import, so these must be set beforehand.
config.SEED = 42
config.N_FOLDS = 1  # Run only 1 fold instead of 5
config.NUM_EPOCHS = 1  # Run only 1 epoch
config.BATCH_SIZE = 32  # Standard batch size
config.PATIENCE = 1  # minimal patience

print("Configuration optimized: N_FOLDS=1, NUM_EPOCHS=1.")

# 2. Import Library Modules
# Note: Importing 'library.train' triggers its main() function automatically.
# We capture stdout to suppress the training logs for this demonstration.
print("Importing library.train (executing full pipeline)...")
buffer = io.StringIO()
original_stdout = sys.stdout
sys.stdout = buffer

try:
    from library import train
    from library import model
    from library import data_loader
except Exception as e:
    sys.stdout = original_stdout
    print("\nError during library import/execution:")
    print(e)
    print("\nCaptured Output:")
    print(buffer.getvalue())
    sys.exit(1)

# Restore stdout
sys.stdout = original_stdout
print("Pipeline execution complete. Proceeding with component verification.\n")


def run_demonstration():
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ==========================================
    # 3. Data Loader Verification
    # ==========================================
    print("\n[1] Verifying Data Loader...")
    # Retrieve loaders using the library function
    train_loader, val_loader, test_loader = data_loader.get_loaders(
        load_cached_data=True
    )

    # Fetch a single batch to inspect
    images, angles, labels = next(iter(train_loader))

    print(
        f"    Batch Shapes -> Images: {images.shape}, Angles: {angles.shape}, Labels: {labels.shape}"
    )

    # Assertions
    if images.shape != (config.BATCH_SIZE, 3, 75, 75):
        raise AssertionError(
            f"Image batch shape mismatch. Expected {(config.BATCH_SIZE, 3, 75, 75)}, got {images.shape}"
        )
    if angles.shape != (config.BATCH_SIZE,):
        raise AssertionError("Angle batch shape mismatch.")
    if labels.shape != (config.BATCH_SIZE,):
        raise AssertionError("Label batch shape mismatch.")

    # Check normalization (approximate bounds)
    if images.max() > 1.0 or images.min() < 0.0:
        raise AssertionError("Images do not appear to be normalized to [0, 1].")

    print("    Data Loader verified successfully.")

    # ==========================================
    # 4. Model Architecture Verification
    # ==========================================
    print("\n[2] Verifying Model (WEBN)...")
    # Instantiate model
    net = model.WEBN().to(device)

    # Prepare inputs
    images_dev = images.to(device)
    angles_dev = angles.to(device)

    # Forward pass
    logits = net(images_dev, angles_dev)

    print(f"    Output Logits Shape: {logits.shape}")

    # Assertions
    if logits.shape != (config.BATCH_SIZE, 1):
        raise AssertionError("Model output shape mismatch. Expected (B, 1).")
    if torch.isnan(logits).any():
        raise AssertionError("Model output contains NaNs.")

    print("    Model architecture verified successfully.")

    # ==========================================
    # 5. Training Logic Verification
    # ==========================================
    print("\n[3] Verifying Training & Validation Steps...")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-4)

    # Execute one training epoch manually
    # train_one_epoch iterates the entire loader, but since we verified the loader works,
    # and the dataset is small, this is acceptable.
    loss, acc = train.train_one_epoch(net, train_loader, criterion, optimizer, device)
    print(f"    Manual Train Epoch -> Loss: {loss:.4f}, Acc: {acc:.4f}")

    if np.isnan(loss) or loss < 0:
        raise AssertionError("Invalid training loss returned.")
    if not (0.0 <= acc <= 1.0):
        raise AssertionError("Training accuracy out of bounds [0, 1].")

    # Execute one validation epoch manually
    val_loss, val_acc = train.validate(net, val_loader, criterion, device)
    print(f"    Manual Val Epoch   -> Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

    if np.isnan(val_loss):
        raise AssertionError("Invalid validation loss returned.")

    print("    Training functions verified successfully.")

    # ==========================================
    # 6. Utility Verification (Early Stopping)
    # ==========================================
    print("\n[4] Verifying Early Stopping Logic...")
    es = utils.EarlyStopping(patience=2, verbose=False)

    # Simulate Step 1: Baseline
    es(1.0, net)
    if es.best_score != -1.0:
        raise AssertionError("EarlyStopping failed to initialize best_score.")

    # Simulate Step 2: Improvement (Loss decreases to 0.9)
    es(0.9, net)
    if es.best_score != -0.9 or es.counter != 0:
        raise AssertionError("EarlyStopping failed to recognize improvement.")

    # Simulate Step 3: Degradation (Loss increases to 0.95)
    es(0.95, net)
    if es.counter != 1:
        raise AssertionError(
            "EarlyStopping failed to increment counter on degradation."
        )

    print("    Early Stopping logic verified successfully.")

    # ==========================================
    # 7. Submission Verification
    # ==========================================
    print("\n[5] Verifying Submission Generation...")
    # The import of library.train should have generated a submission file
    if not os.path.exists(config.SUBMISSION_PATH):
        raise AssertionError(f"Submission file not found at {config.SUBMISSION_PATH}")

    df_sub = pd.read_csv(config.SUBMISSION_PATH)
    print(f"    Submission File: {config.SUBMISSION_PATH}")
    print(f"    Rows: {len(df_sub)}")
    print(f"    Columns: {list(df_sub.columns)}")

    if "id" not in df_sub.columns or "is_iceberg" not in df_sub.columns:
        raise AssertionError("Submission file missing required columns.")

    # Check probability range
    preds = df_sub["is_iceberg"].values
    if preds.min() < 0 or preds.max() > 1:
        raise AssertionError("Submission probabilities out of range [0, 1].")

    print("    Submission file verified successfully.")
    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    run_demonstration()
