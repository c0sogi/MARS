import os
import sys
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import shutil

# Import library modules
from library.config import Config
from library.data_handler import get_dataloaders
from library.architecture import MSCWDSModel
from library.engine import Trainer, generate_submission
from library.utils import transform_targets, inverse_transform_targets, StandardScaler


def set_seed(seed=42):
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    print("Initializing Demonstration...")

    # 1. Configure for Speed
    # We override Config parameters to run a fast demo on a subset of data
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = "./working/demo_submission"

    # Use separate cache files for demo to avoid messing with full training artifacts
    Config.TRAIN_DATA_CACHE = os.path.join(Config.WORKING_DIR, "train_data.npz")
    Config.VAL_DATA_CACHE = os.path.join(Config.WORKING_DIR, "val_data.npz")
    Config.TEST_DATA_CACHE = os.path.join(Config.WORKING_DIR, "test_data.npz")
    Config.SCALERS_CACHE = os.path.join(Config.WORKING_DIR, "scalers.npz")
    Config.MODEL_CHECKPOINT = os.path.join(Config.WORKING_DIR, "best_model.pt")
    Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_DIR, "demo_submission.csv")

    # Hyperparameters for demo
    Config.DEBUG_SAMPLE_SIZE = 100  # Use only 100 samples
    Config.BATCH_SIZE = 16
    Config.NUM_EPOCHS = 2
    Config.PATIENCE = 2
    Config.ATOMIC_HIDDEN_DIM = 64  # Smaller model for speed
    Config.GLOBAL_HIDDEN_DIM = 32

    Config.ensure_directories()
    set_seed(Config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Verify Utility Functions
    print("\nVerifying Utilities...")
    dummy_targets = np.array([0.0, 1.0, 10.0])
    transformed = transform_targets(dummy_targets)
    inverted = inverse_transform_targets(transformed)
    assert np.allclose(
        dummy_targets, inverted
    ), "Target transformation is not reversible!"
    print("Target transformation logic verified.")

    scaler = StandardScaler()
    dummy_data = np.random.rand(10, 5)
    scaled_data = scaler.fit_transform(dummy_data)
    unscaled_data = scaler.inverse_transform(scaled_data)
    assert np.allclose(dummy_data, unscaled_data), "StandardScaler logic incorrect!"
    print("StandardScaler logic verified.")

    # 3. Data Loading
    print("\nLoading Data...")
    # This will trigger computation of features since cache paths are new
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        load_cached_data=True,  # Will fail first time and compute
        sample_size=Config.DEBUG_SAMPLE_SIZE,
    )

    # Inspect a single batch
    print("\nInspecting Train Batch...")
    batch = next(iter(train_loader))

    atomic_feats = batch["atomic_features"]
    global_feats = batch["global_features"]
    targets = batch["targets"]
    batch_idx = batch["batch_index"]
    ids = batch["ids"]

    print(f"Atomic Features Shape: {atomic_feats.shape}")  # (Total_Atoms, 16)
    print(f"Global Features Shape: {global_feats.shape}")  # (Batch_Size, 12)
    print(f"Targets Shape: {targets.shape}")  # (Batch_Size, 2)
    print(f"Batch Index Shape: {batch_idx.shape}")  # (Total_Atoms,)

    assert (
        atomic_feats.shape[1] == Config.ATOMIC_INPUT_DIM
    ), "Incorrect atomic feature dim"
    assert (
        global_feats.shape[1] == Config.GLOBAL_INPUT_DIM
    ), "Incorrect global feature dim"
    assert targets.shape[1] == Config.NUM_TARGETS, "Incorrect target dim"
    assert atomic_feats.shape[0] == batch_idx.shape[0], "Batch index size mismatch"
    assert global_feats.shape[0] == len(ids), "Batch size mismatch with IDs"

    # 4. Model Initialization
    print("\nInitializing Model...")
    model = MSCWDSModel()
    model.to(device)

    # Test Forward Pass
    print("Testing Forward Pass...")
    with torch.no_grad():
        # Move batch to device
        inputs = {
            "atomic_features": atomic_feats.to(device),
            "batch_index": batch_idx.to(device),
            "global_features": global_feats.to(device),
        }
        outputs = model(inputs)
        print(f"Output Shape: {outputs.shape}")
        assert outputs.shape == targets.shape, "Model output shape mismatch"

    # 5. Training
    print("\nStarting Training Loop...")
    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
    )

    trainer = Trainer(model, optimizer, scheduler, device)

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=Config.NUM_EPOCHS,
        patience=Config.PATIENCE,
        checkpoint_path=Config.MODEL_CHECKPOINT,
    )

    # Verify checkpoint creation
    if not os.path.exists(Config.MODEL_CHECKPOINT):
        # If training was too short or loss didn't improve (unlikely with random init), save manually for demo
        print("Saving manual checkpoint for demo continuity...")
        torch.save(model.state_dict(), Config.MODEL_CHECKPOINT)

    assert os.path.exists(
        Config.MODEL_CHECKPOINT
    ), "Model checkpoint not found after training!"

    # 6. Submission Generation
    print("\nGenerating Submission...")
    generate_submission(
        model=model,
        test_loader=test_loader,
        device=device,
        output_path=Config.SUBMISSION_FILE,
    )

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file not created"

    sub_df = pd.read_csv(Config.SUBMISSION_FILE)
    print(f"Submission shape: {sub_df.shape}")
    print("Head of submission:")
    print(sub_df.head())

    # Check columns
    expected_cols = ["id"] + Config.TARGET_COLS
    assert (
        list(sub_df.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(sub_df.columns)}"

    # Check if we have rows (should be equal to test sample size or full test size if not sampled)
    # Since we set DEBUG_SAMPLE_SIZE=100, we expect min(100, total_test_samples) rows
    expected_len = min(
        Config.DEBUG_SAMPLE_SIZE, 240
    )  # 240 is total test size known from metadata
    assert (
        len(sub_df) == expected_len
    ), f"Expected {expected_len} predictions, got {len(sub_df)}"

    print("\nDemonstration Completed Successfully!")


if __name__ == "__main__":
    main()
