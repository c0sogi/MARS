import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, get_optimizer_grouped_parameters
from library.dataset import get_dataloaders
from library.model import HybridSwiGLUNet
from library.engine import train_one_epoch, evaluate, predict


def run_demonstration():
    print("--- Starting Demonstration Script ---")

    # 1. Setup and Configuration Overrides for Speed
    # We override Config attributes to make the model smaller and the run faster.
    print("1. Configuring environment and overriding Config for speed...")
    seed_everything(Config.SEED)

    # Reduce Model Complexity for demo purposes
    Config.EMBED_DIM = 16
    Config.TRANSFORMER_LAYERS = 1
    Config.TRANSFORMER_HEADS = 2
    Config.BACKBONE_STAGES = [32, 16]  # Tiny backbone
    Config.BLOCKS_PER_STAGE = 1

    # Reduce Training params
    Config.BATCH_SIZE = 128
    Config.EPOCHS = 1

    # Ensure device is set correctly
    device = torch.device(Config.DEVICE)
    print(f"   Device: {device}")
    print("   Config overrides applied.")

    # 2. Data Loading
    print("\n2. Loading Data...")
    # This will load the full dataset into memory (cached if available)
    train_loader_full, val_loader_full, test_loader_full, test_ids = get_dataloaders(
        load_cached_data=True
    )

    # Verify Data Integrity
    sample_batch = next(iter(train_loader_full))
    continuous = sample_batch["continuous"]
    categorical = sample_batch["categorical"]
    targets = sample_batch["target"]

    print(
        f"   Batch shapes - Continuous: {continuous.shape}, Categorical: {categorical.shape}, Targets: {targets.shape}"
    )

    # Assertions for data structure
    if continuous.shape[1] != Config.NUM_CONTINUOUS_FEATURES:
        raise AssertionError(
            f"Expected {Config.NUM_CONTINUOUS_FEATURES} continuous features, got {continuous.shape[1]}"
        )
    if categorical.shape[1] != Config.SEQUENCE_LENGTH:
        raise AssertionError(
            f"Expected sequence length {Config.SEQUENCE_LENGTH}, got {categorical.shape[1]}"
        )
    if targets.ndim != 1:
        raise AssertionError("Targets should be 1D tensor")

    print("   Data loaded and verified.")

    # 3. Model Instantiation & Verification
    print("\n3. Instantiating Model...")
    model = HybridSwiGLUNet().to(device)

    # Dummy Forward Pass to check connectivity and output shapes
    print("   Performing dummy forward pass...")
    with torch.no_grad():
        dummy_cont = continuous.to(device)
        dummy_cat = categorical.to(device)
        output = model(dummy_cont, dummy_cat)

    # Assertions for model output
    if output.shape != (Config.BATCH_SIZE, 1):
        raise AssertionError(
            f"Expected output shape {(Config.BATCH_SIZE, 1)}, got {output.shape}"
        )
    if output.min() < 0.0 or output.max() > 1.0:
        raise AssertionError("Model output (sigmoid) is out of range [0, 1]")

    print("   Model instantiated and forward pass verified.")

    # 4. Training Loop Demonstration (Subset)
    print("\n4. Demonstrating Training Loop (Subset)...")

    # Create a small subset for training demonstration (e.g., 5 batches)
    # We use a Subset and wrap it in a new DataLoader so train_one_epoch works correctly
    subset_size = Config.BATCH_SIZE * 5
    train_subset = Subset(train_loader_full.dataset, range(subset_size))
    train_loader_subset = DataLoader(
        train_subset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        drop_last=True,  # Important for BatchNorm stability
    )

    # Setup Optimizer
    param_groups = get_optimizer_grouped_parameters(model)
    optimizer = torch.optim.AdamW(param_groups, lr=Config.LEARNING_RATE)
    criterion = nn.BCELoss()

    # Run one epoch on subset
    train_loss = train_one_epoch(
        model, train_loader_subset, optimizer, criterion, device
    )

    print(f"   Subset Train Loss: {train_loss:.4f}")
    if not np.isfinite(train_loss):
        raise AssertionError("Training loss is not finite.")

    # 5. Evaluation Demonstration (Subset)
    print("\n5. Demonstrating Evaluation (Subset)...")

    val_subset_size = Config.BATCH_SIZE * 5
    val_subset = Subset(val_loader_full.dataset, range(val_subset_size))
    val_loader_subset = DataLoader(
        val_subset, batch_size=Config.BATCH_SIZE, shuffle=False
    )

    val_loss, val_auc = evaluate(model, val_loader_subset, criterion, device)

    print(f"   Subset Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")
    if not (0.0 <= val_auc <= 1.0):
        raise AssertionError("AUC score is out of valid range [0, 1]")

    # 6. Inference Demonstration (Subset)
    print("\n6. Demonstrating Inference (Subset)...")

    test_subset_size = Config.BATCH_SIZE * 2
    test_subset = Subset(test_loader_full.dataset, range(test_subset_size))
    test_loader_subset = DataLoader(
        test_subset, batch_size=Config.BATCH_SIZE, shuffle=False
    )

    predictions = predict(model, test_loader_subset, device)

    print(f"   Predictions shape: {predictions.shape}")
    if len(predictions) != test_subset_size:
        raise AssertionError(
            f"Expected {test_subset_size} predictions, got {len(predictions)}"
        )

    # 7. Generate Submission
    print("\n7. Generating Dummy Submission...")
    # Using the subset IDs corresponding to the predictions
    subset_test_ids = test_ids[:test_subset_size]

    submission_df = pd.DataFrame({"id": subset_test_ids, "target": predictions})

    # Save to working directory
    output_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    submission_df.to_csv(output_path, index=False)

    print(f"   Submission saved to {output_path}")

    # Final check of the file
    saved_df = pd.read_csv(output_path)
    if saved_df.shape != (test_subset_size, 2):
        raise AssertionError("Saved submission file has incorrect shape.")

    print("\n--- Demonstration Complete Successfully ---")


if __name__ == "__main__":
    run_demonstration()
