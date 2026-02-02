import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, get_device
from library.data_loader import load_and_preprocess_data, ManufacturingDataset
from library.model import ModalityScaledHybridSwiGLU
from library.train_eval import get_optimizer, train_one_epoch, validate, predict


def main():
    print("=== Starting Demonstration Script ===")

    # --------------------------------------------------------------------------
    # 1. Configuration Override for Speed and Isolation
    # --------------------------------------------------------------------------
    print("\n[1] Configuring environment...")

    # Set a specific working directory for this demo to avoid overwriting main artifacts
    demo_dir = "./working/demo_script_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Override Config class attributes directly
    Config.WORKING_DIR = demo_dir
    Config.CACHE_PATH = os.path.join(demo_dir, "processed_data.npz")
    Config.MODEL_PATH = os.path.join(demo_dir, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")

    # Reduce training duration for demonstration
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 2048  # Large batch size for speed on A100

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Epochs: {Config.EPOCHS}")

    # --------------------------------------------------------------------------
    # 2. Verify Utilities
    # --------------------------------------------------------------------------
    print("\n[2] Verifying Utilities...")

    # Test Reproducibility
    set_seed(Config.SEED)
    rand1 = np.random.rand()
    set_seed(Config.SEED)
    rand2 = np.random.rand()

    assert rand1 == rand2, "set_seed did not ensure reproducibility for NumPy."
    print("Reproducibility check passed.")

    # Test Device
    device = get_device()
    print(f"Device selected: {device}")

    # --------------------------------------------------------------------------
    # 3. Verify Data Loading
    # --------------------------------------------------------------------------
    print("\n[3] Verifying Data Loading...")

    # This will process data from scratch or load from cache if available in the new dir
    train_loader, val_loader, test_loader = load_and_preprocess_data(
        load_cached_data=True, batch_size=Config.BATCH_SIZE
    )

    # Verify Train Loader
    batch = next(iter(train_loader))
    continuous = batch["continuous"]
    sequence = batch["sequence"]
    targets = batch["target"]

    # Assertions
    assert continuous.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CONTINUOUS,
    ), f"Continuous shape mismatch: {continuous.shape}"
    assert sequence.shape == (
        Config.BATCH_SIZE,
        Config.SEQUENCE_LENGTH,
    ), f"Sequence shape mismatch: {sequence.shape}"
    assert targets.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Target shape mismatch: {targets.shape}"
    assert sequence.dtype == torch.long, "Sequence data must be LongTensor"
    assert continuous.dtype == torch.float32, "Continuous data must be FloatTensor"

    print(
        f"Data Batch Verified. Continuous: {continuous.shape}, Sequence: {sequence.shape}"
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # --------------------------------------------------------------------------
    # 4. Verify Model Architecture
    # --------------------------------------------------------------------------
    print("\n[4] Verifying Model Architecture...")

    model = ModalityScaledHybridSwiGLU().to(device)

    # Move batch to device
    cont_dev = continuous.to(device)
    seq_dev = sequence.to(device)

    # Forward pass
    logits = model(cont_dev, seq_dev)

    # Assertions
    assert logits.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Model output shape mismatch. Expected {(Config.BATCH_SIZE, 1)}, got {logits.shape}"
    assert not torch.isnan(logits).any(), "Model output contains NaNs"

    print("Model forward pass successful. Output shape verified.")

    # --------------------------------------------------------------------------
    # 5. Verify Optimizer Construction
    # --------------------------------------------------------------------------
    print("\n[5] Verifying Optimizer...")

    optimizer = get_optimizer(model)

    # Check parameter groups
    assert len(optimizer.param_groups) == 2, "Optimizer should have 2 parameter groups"

    # Group 0: Decay, Group 1: No Decay
    decay_group = optimizer.param_groups[0]
    no_decay_group = optimizer.param_groups[1]

    assert (
        decay_group["weight_decay"] == Config.WEIGHT_DECAY_GROUP1
    ), "Decay group incorrect"
    assert (
        no_decay_group["weight_decay"] == Config.WEIGHT_DECAY_GROUP2
    ), "No-decay group incorrect"

    print(
        f"Optimizer groups verified. Decay: {decay_group['weight_decay']}, No Decay: {no_decay_group['weight_decay']}"
    )

    # --------------------------------------------------------------------------
    # 6. Execution Loop (Train, Validate, Predict)
    # --------------------------------------------------------------------------
    print("\n[6] Running Training Loop (1 Epoch)...")

    criterion = torch.nn.BCEWithLogitsLoss()

    # Train
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    print(f"Training completed. Loss: {train_loss:.4f}")
    assert train_loss > 0, "Training loss should be positive"

    # Validate
    val_loss, val_auc = validate(model, val_loader, criterion, device)
    print(f"Validation completed. Loss: {val_loss:.4f}, AUC: {val_auc:.4f}")
    assert 0 <= val_auc <= 1, "AUC score out of range"

    # Save Model (simulating checkpointing)
    torch.save(model.state_dict(), Config.MODEL_PATH)
    assert os.path.exists(Config.MODEL_PATH), "Model file was not saved"
    print("Model checkpoint saved.")

    # Predict
    print("Generating predictions on test set...")
    predictions = predict(model, test_loader, device)

    assert len(predictions) == len(
        test_loader.dataset
    ), f"Prediction count mismatch. Expected {len(test_loader.dataset)}, got {len(predictions)}"
    assert (
        predictions.min() >= 0 and predictions.max() <= 1
    ), "Predictions should be probabilities between 0 and 1"

    print(f"Predictions generated. Shape: {predictions.shape}")

    # --------------------------------------------------------------------------
    # 7. Submission Generation
    # --------------------------------------------------------------------------
    print("\n[7] Generating Submission File...")

    # Load test metadata to align IDs
    test_meta = pd.read_csv(Config.TEST_META_PATH)

    # Create submission DataFrame
    submission_df = pd.DataFrame(
        {"id": test_meta["id"], "target": predictions.flatten()}
    )

    # Save
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    # Verify file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created"

    # Check format
    df_check = pd.read_csv(Config.SUBMISSION_PATH)
    assert list(df_check.columns) == ["id", "target"], "Submission columns mismatch"
    assert len(df_check) == len(test_meta), "Submission row count mismatch"

    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
