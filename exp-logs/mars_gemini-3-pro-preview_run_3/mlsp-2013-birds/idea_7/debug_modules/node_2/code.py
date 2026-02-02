import os
import torch
import pandas as pd
import numpy as np
import sys

# Import provided library modules
from library.config import Config
from library.utils import seed_everything
from library.dataset import get_dataloaders
from library.model import MILResNet18
from library.trainer import run_fold


def main():
    print("Starting Bird Species Classification Demo...")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # Override Config parameters for a fast demonstration
    print("Configuring for demo mode...")
    Config.EPOCHS = 1
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 16  # Small subset for speed
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for demo
    Config.N_FOLDS = 2  # We only need to run one fold

    # Ensure reproducibility
    seed_everything(Config.SEED)

    # Ensure directories exist
    Config.create_dirs()

    # -------------------------------------------------------------------------
    # 2. Data Loading & Processing
    # -------------------------------------------------------------------------
    print("\n[Step 1] Generating features and creating DataLoaders...")

    # We set load_cached_data=False to demonstrate the generation logic from raw audio.
    # This will process the audio files, create spectrograms, and save them to cache.
    train_loader, val_loader, test_loader = get_dataloaders(
        fold=0, load_cached_data=False
    )

    # Verify DataLoaders
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Fetch a single batch to verify shapes
    try:
        inputs, targets = next(iter(train_loader))
    except StopIteration:
        raise RuntimeError(
            "Train loader is empty! Check dataset paths or debug subset size."
        )

    print(f"Input batch shape: {inputs.shape}")  # Expected: (B, Tiles, C, H, W)
    print(f"Target batch shape: {targets.shape}")  # Expected: (B, Num_Classes)

    # Assertions to verify data integrity
    # Inputs: (Batch, Num_Tiles, Channels, Height, Width)
    assert inputs.ndim == 5, f"Expected 5D input tensor, got {inputs.ndim}D"
    assert (
        inputs.shape[0] == Config.BATCH_SIZE
    ), f"Batch size mismatch. Got {inputs.shape[0]}"
    assert (
        inputs.shape[1] == Config.NUM_TILES
    ), f"Tile count mismatch. Got {inputs.shape[1]}"
    assert inputs.shape[2] == 3, f"Channel count mismatch. Got {inputs.shape[2]}"
    assert inputs.shape[3] == Config.IMG_SIZE[0], "Image height mismatch"
    assert inputs.shape[4] == Config.IMG_SIZE[1], "Image width mismatch"

    # Targets: (Batch, Num_Classes)
    assert (
        targets.shape[1] == Config.NUM_CLASSES
    ), f"Class count mismatch. Got {targets.shape[1]}"

    print("Data shapes verified successfully.")

    # -------------------------------------------------------------------------
    # 3. Model Instantiation & Forward Pass
    # -------------------------------------------------------------------------
    print("\n[Step 2] Initializing Model and testing forward pass...")

    model = MILResNet18()
    model.to(Config.DEVICE)
    model.eval()

    with torch.no_grad():
        # Move inputs to device
        inputs_device = inputs.to(Config.DEVICE)
        outputs = model(inputs_device)

    print(f"Model output shape: {outputs.shape}")

    # Assertions
    assert outputs.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), "Model output shape mismatch"
    # Check that outputs are logits (not probabilities), so they can be negative or > 1
    # Just a sanity check that it's not crashing.

    print("Model forward pass successful.")

    # -------------------------------------------------------------------------
    # 4. Training Loop (Single Fold)
    # -------------------------------------------------------------------------
    print("\n[Step 3] Running Training Loop for Fold 0...")

    # run_fold handles the training loop, validation, and checkpoint saving.
    # We use load_cached_data=True here because we generated data in Step 1.
    best_auc = run_fold(fold=0, load_cached_data=True)

    print(f"Training completed. Best Validation AUC: {best_auc:.4f}")

    # Verify checkpoint exists
    checkpoint_path = os.path.join(Config.CHECKPOINTS_DIR, "fold_0_best.pth")
    assert os.path.exists(
        checkpoint_path
    ), f"Checkpoint file not found at {checkpoint_path}"
    print(f"Checkpoint verified at {checkpoint_path}")

    # -------------------------------------------------------------------------
    # 5. Inference & Submission Generation
    # -------------------------------------------------------------------------
    print("\n[Step 4] Running Inference on Test Set and generating submission...")

    # Load the best model
    model.load_state_dict(torch.load(checkpoint_path, map_location=Config.DEVICE))
    model.eval()

    predictions = []

    # Iterate over test loader
    with torch.no_grad():
        for batch_inputs, _ in test_loader:
            batch_inputs = batch_inputs.to(Config.DEVICE)
            logits = model(batch_inputs)
            probs = torch.sigmoid(logits)
            predictions.append(probs.cpu().numpy())

    all_probs = np.concatenate(predictions, axis=0)

    # Get Test DataFrame to map IDs
    test_df = test_loader.dataset.df

    assert len(all_probs) == len(
        test_df
    ), "Mismatch between predictions and test set size"

    # Format submission: Id, Probability
    # Id = rec_id * 100 + species_id
    submission_data = []

    for idx, row in test_df.iterrows():
        rec_id = row["rec_id"]
        row_probs = all_probs[idx]

        for species_id in range(Config.NUM_CLASSES):
            prob = row_probs[species_id]
            submission_id = int(rec_id * 100 + species_id)
            submission_data.append({"Id": submission_id, "Probability": prob})

    submission_df = pd.DataFrame(submission_data)

    # Verify submission format
    assert "Id" in submission_df.columns
    assert "Probability" in submission_df.columns
    assert len(submission_df) == len(test_df) * Config.NUM_CLASSES

    # Save submission
    out_file = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(out_file, index=False)

    print(f"Submission saved to {out_file}")
    print("First 5 rows:")
    print(submission_df.head())

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()
