import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Import from the provided library
from library.config import Config
from library.data_utils import preprocess_pipeline
from library.dataset import create_dataloaders
from library.model import ManufacturingTransformer
from library.train_eval import train_one_epoch, evaluate, predict, set_seed


def main():
    print("Starting demonstration of Manufacturing Control library...")

    # -------------------------------------------------------------------------
    # 1. Configuration Setup for Demo
    # -------------------------------------------------------------------------
    # We override specific Config attributes to ensure the demo runs quickly
    # and uses a separate working directory.
    print("\n[1] Configuring environment for fast demonstration...")

    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    Config.MODEL_CHECKPOINT_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")

    # Reduce training duration for demo
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 128  # Smaller batch size for the small subset

    # Re-run setup to create the new directories
    Config.setup()
    set_seed(Config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Device: {device}")
    print(f"    Working Directory: {Config.WORKING_DIR}")

    # -------------------------------------------------------------------------
    # 2. Data Preprocessing
    # -------------------------------------------------------------------------
    print("\n[2] Running Data Preprocessing Pipeline...")

    # Force reprocessing to demonstrate the logic (load_cached_data=False)
    # This reads from ./metadata/*.csv and saves .npy files to the new cache dir
    data_dict, vocab_size = preprocess_pipeline(load_cached_data=False)

    # Logic Verification: Check Data Integrity
    print("    Verifying processed data shapes...")

    # Check keys
    expected_keys = [
        "X_num_train",
        "X_seq_train",
        "y_train",
        "X_num_val",
        "X_seq_val",
        "y_val",
        "X_num_test",
        "X_seq_test",
        "ids_test",
    ]
    for key in expected_keys:
        if key not in data_dict:
            raise AssertionError(f"Missing key in data_dict: {key}")

    # Check shapes
    n_train = data_dict["X_num_train"].shape[0]
    n_val = data_dict["X_num_val"].shape[0]
    n_test = data_dict["X_num_test"].shape[0]

    # Numerical features should match across splits
    num_feats = data_dict["X_num_train"].shape[1]
    if data_dict["X_num_val"].shape[1] != num_feats:
        raise AssertionError(
            "Mismatch in numerical feature count between train and val."
        )

    # Sequence length should be consistent (Config.SEQ_LEN is 10)
    seq_len = data_dict["X_seq_train"].shape[1]
    if seq_len != Config.SEQ_LEN:
        raise AssertionError(f"Sequence length is {seq_len}, expected {Config.SEQ_LEN}")

    print(f"    Vocab Size: {vocab_size}")
    print(f"    Train samples: {n_train}, Val samples: {n_val}, Test samples: {n_test}")
    print(f"    Numerical Features: {num_feats}")

    # -------------------------------------------------------------------------
    # 3. DataLoader Creation (with Subset)
    # -------------------------------------------------------------------------
    print("\n[3] Creating DataLoaders (Subset)...")

    # We limit samples to 1000 to ensure the training loop runs instantly
    subset_size = 1000
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dict,
        batch_size=Config.BATCH_SIZE,
        num_workers=0,  # Set to 0 for simple main-process loading in demo
        pin_memory=False,
        limit_samples=subset_size,
    )

    # Logic Verification: Check Batch Structure
    print("    Verifying batch structure...")
    sample_batch = next(iter(train_loader))

    if (
        "numerical" not in sample_batch
        or "sequence" not in sample_batch
        or "target" not in sample_batch
    ):
        raise AssertionError("Batch missing required keys.")

    b_num = sample_batch["numerical"]
    b_seq = sample_batch["sequence"]
    b_tgt = sample_batch["target"]

    if b_num.shape != (Config.BATCH_SIZE, num_feats):
        raise AssertionError(f"Incorrect numerical batch shape: {b_num.shape}")
    if b_seq.shape != (Config.BATCH_SIZE, seq_len):
        raise AssertionError(f"Incorrect sequence batch shape: {b_seq.shape}")
    if b_tgt.shape != (Config.BATCH_SIZE, 1):
        raise AssertionError(f"Incorrect target batch shape: {b_tgt.shape}")

    print("    Batch structure verified.")

    # -------------------------------------------------------------------------
    # 4. Model Instantiation and Forward Pass
    # -------------------------------------------------------------------------
    print("\n[4] Instantiating Model and testing Forward Pass...")

    model = ManufacturingTransformer(
        num_numerical_features=num_feats, vocab_size=vocab_size, seq_len=seq_len
    ).to(device)

    # Move sample batch to device
    x_num_dev = b_num.to(device)
    x_seq_dev = b_seq.to(device)

    # Forward pass
    logits = model(x_num_dev, x_seq_dev)

    # Logic Verification: Output Shape
    if logits.shape != (Config.BATCH_SIZE, 1):
        raise AssertionError(
            f"Model output shape mismatch. Expected ({Config.BATCH_SIZE}, 1), got {logits.shape}"
        )

    print("    Forward pass successful. Output shape verified.")

    # -------------------------------------------------------------------------
    # 5. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n[5] Running Training Loop (2 Epochs on subset)...")

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = None  # Skip scheduler for simple demo

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device
        )
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)

        print(
            f"    Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val AUC={val_auc:.4f}"
        )

        # Logic Verification: Metrics
        if np.isnan(train_loss) or np.isnan(val_loss):
            raise AssertionError("Loss is NaN.")
        if not (0 <= val_auc <= 1):
            raise AssertionError(f"Invalid AUC score: {val_auc}")

    # Save the model (simulating checkpointing)
    torch.save(model.state_dict(), Config.MODEL_CHECKPOINT_PATH)
    if not os.path.exists(Config.MODEL_CHECKPOINT_PATH):
        raise AssertionError("Model checkpoint file was not created.")
    print("    Training complete and model saved.")

    # -------------------------------------------------------------------------
    # 6. Prediction and Submission
    # -------------------------------------------------------------------------
    print("\n[6] Generating Predictions and Submission...")

    # Load model
    model.load_state_dict(torch.load(Config.MODEL_CHECKPOINT_PATH, map_location=device))

    # Predict on test subset
    predictions = predict(model, test_loader, device)

    # Logic Verification: Prediction Shape
    expected_preds = min(subset_size, n_test)  # We limited samples
    if len(predictions) != expected_preds:
        raise AssertionError(
            f"Prediction count mismatch. Expected {expected_preds}, got {len(predictions)}"
        )

    # Create Submission
    # Get corresponding IDs (sliced same as data)
    ids = data_dict["ids_test"][:expected_preds]

    submission = pd.DataFrame({"id": ids, "target": predictions})

    submission.to_csv(Config.SUBMISSION_PATH, index=False)

    # Logic Verification: File Existence and Format
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise AssertionError("Submission file not found.")

    df_check = pd.read_csv(Config.SUBMISSION_PATH)
    if list(df_check.columns) != ["id", "target"]:
        raise AssertionError("Submission columns mismatch.")
    if len(df_check) != expected_preds:
        raise AssertionError("Submission row count mismatch.")

    print(f"    Submission saved to {Config.SUBMISSION_PATH}")
    print(f"    First 3 rows:\n{df_check.head(3)}")

    print("\nDemonstration completed successfully!")


if __name__ == "__main__":
    main()
