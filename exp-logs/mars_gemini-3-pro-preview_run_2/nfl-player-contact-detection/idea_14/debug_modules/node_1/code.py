import os
import sys
import numpy as np
import pandas as pd
import torch
import shutil


# --- 1. Suppress Progress Bars (Monkey Patching) ---
# The prompt requires no progress bars. The provided library uses tqdm.
# We patch it before importing the library modules.
def silent_tqdm(iterable, *args, **kwargs):
    return iterable


import tqdm

tqdm.tqdm = silent_tqdm

# --- Import Library Modules ---
from library.config import Config
from library.data_processing import DataProcessor
from library.dataset import get_dataloaders, get_test_loader
from library.model import ECGRN
from library.loss import FocalLoss
from library.train_eval import train_model, set_seed


def main():
    print("--- Starting NFL Contact Detection Demonstration ---")

    # --- 2. Configuration ---
    # Initialize Config with debug settings for speed.
    # sample_size=2000 ensures we have enough data to form batches but runs fast.
    # epochs=1 ensures the training loop finishes quickly.
    config = Config(debug=True, sample_size=2000, epochs=1)

    # Force a clean state for demonstration by removing cached files if they exist
    # This ensures we actually run the processing logic.
    if os.path.exists(config.WORKING_DIR):
        shutil.rmtree(config.WORKING_DIR)
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    config.display()

    # Set global seeds for reproducibility
    set_seed(config.SEED)

    # --- 3. Data Processing Verification ---
    print("\n[Step 1] Verifying Data Processing...")
    processor = DataProcessor(config)

    # Load and process training/validation data
    # This triggers _process_pipeline -> _generate_wide_features -> _impute_ground... -> _prepare_tensors
    X_train, X_cat_train, y_train, X_val, X_cat_val, y_val = (
        processor.load_and_process_train_val(load_cached_data=False)
    )

    # Assertions
    print(
        f"  Train shapes: Cont={X_train.shape}, Cat={X_cat_train.shape}, Target={y_train.shape}"
    )

    # Check Continuous Features
    # Shape: (N, num_continuous). num_continuous depends on window size (5) -> 11 steps * 16 features approx
    assert X_train.ndim == 2, "X_train must be 2D"
    assert X_train.dtype == np.float32, "X_train must be float32"

    # Check Categorical Features
    # Shape: (N, 4) -> [pos1, team1, pos2, team2]
    assert X_cat_train.ndim == 2, "X_cat_train must be 2D"
    assert X_cat_train.shape[1] == 4, "X_cat_train must have 4 columns"
    assert np.issubdtype(X_cat_train.dtype, np.integer), "X_cat_train must be integer"

    # Check Targets
    assert y_train.ndim == 1, "y_train must be 1D array"
    assert set(np.unique(y_train)).issubset({0.0, 1.0}), "Targets must be binary"

    print("  Data Processing Logic Verified.")

    # --- 4. Dataset & DataLoader Verification ---
    print("\n[Step 2] Verifying Dataset and DataLoader...")
    train_loader, val_loader = get_dataloaders(config, processor)

    # Fetch one batch
    batch_cont, batch_cat, batch_y = next(iter(train_loader))

    print(
        f"  Batch shapes: Cont={batch_cont.shape}, Cat={batch_cat.shape}, Target={batch_y.shape}"
    )

    # Assertions
    assert batch_cont.shape[0] == config.BATCH_SIZE, "Batch size mismatch"
    assert torch.is_tensor(batch_cont), "Output must be a Tensor"
    assert batch_y.shape == (
        config.BATCH_SIZE,
        1,
    ), "Target batch shape must be (Batch, 1)"

    print("  DataLoader Logic Verified.")

    # --- 5. Model & Loss Verification ---
    print("\n[Step 3] Verifying Model Architecture and Loss...")

    num_continuous = batch_cont.shape[1]
    cat_embedding_dims = [
        config.EMBEDDING_DIMS["position"],
        config.EMBEDDING_DIMS["team"],
        config.EMBEDDING_DIMS["position"],
        config.EMBEDDING_DIMS["team"],
    ]

    model = ECGRN(
        num_continuous=num_continuous,
        categorical_embedding_dims=cat_embedding_dims,
        hidden_size=config.HIDDEN_SIZE,
        num_blocks=1,  # Reduced blocks for demo speed
        dropout_rate=config.DROPOUT,
    ).to(config.DEVICE)

    # Move batch to device
    batch_cont = batch_cont.to(config.DEVICE)
    batch_cat = batch_cat.to(config.DEVICE)
    batch_y = batch_y.to(config.DEVICE)

    # Forward Pass
    logits = model(batch_cont, batch_cat)

    # Assertions
    assert logits.shape == (config.BATCH_SIZE, 1), "Logits shape mismatch"

    # Loss Calculation
    criterion = FocalLoss(alpha=config.FOCAL_ALPHA, gamma=config.FOCAL_GAMMA)
    loss = criterion(logits, batch_y)

    # Assertions
    assert loss.dim() == 0, "Loss must be a scalar"
    assert loss.item() >= 0, "Loss must be non-negative"

    print(f"  Forward pass successful. Initial Loss: {loss.item():.4f}")
    print("  Model and Loss Logic Verified.")

    # --- 6. Training Pipeline Verification ---
    print("\n[Step 4] Running Training Loop...")

    # We use the train_model function which handles the loop, validation, and threshold optimization
    # Note: train_model re-initializes the model, so we pass the config and processor.
    trained_model, best_threshold = train_model(config, processor)

    print(f"  Training complete. Best Threshold: {best_threshold}")
    assert isinstance(
        trained_model, torch.nn.Module
    ), "train_model did not return a Module"
    assert 0.0 < best_threshold < 1.0, "Threshold out of bounds"

    # --- 7. Inference & Submission Verification ---
    print("\n[Step 5] Running Inference and Generating Submission...")

    test_loader, test_ids = get_test_loader(config, processor)

    trained_model.eval()
    all_probs = []

    with torch.no_grad():
        for X_cont, X_cat in test_loader:
            X_cont = X_cont.to(config.DEVICE)
            X_cat = X_cat.to(config.DEVICE)

            logits = trained_model(X_cont, X_cat)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.extend(probs)

    all_probs = np.array(all_probs)

    # Verify counts
    assert len(all_probs) == len(
        test_ids
    ), f"Prediction count mismatch: {len(all_probs)} vs {len(test_ids)}"

    # Create Submission DataFrame
    # Apply threshold
    predictions = (all_probs > best_threshold).astype(int)

    submission = pd.DataFrame({"contact_id": test_ids, "contact": predictions})

    # Verify Submission Format
    print(f"  Submission Head:\n{submission.head()}")
    assert "contact_id" in submission.columns
    assert "contact" in submission.columns
    assert submission["contact"].dtype == int or submission["contact"].dtype == np.int64

    # Save (Optional, just to verify write access to working dir)
    sub_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
    submission.to_csv(sub_path, index=False)
    print(f"  Submission saved to {sub_path}")

    print("\n--- Demonstration Complete: All Systems Go ---")


if __name__ == "__main__":
    main()
