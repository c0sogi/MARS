import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import time

from library.config import (
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    WORKING_DIR,
    SUBMISSION_PATH,
    DEVICE,
    SEED,
    NUM_WORKERS,
)
from library.utils import seed_everything
from library.dataset import IceCubeDataset
from library.model import (
    GeometricPulseAggregator,
    CosineDistanceLoss,
    calculate_angular_error,
)
from library.train import train_one_epoch, validate
from library.inference import predict_test_set

# =============================================================================
# Hyperparameters & Constants
# =============================================================================
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
EPOCHS = 5
TRAIN_SUBSET_SIZE = 300_000  # Limit training data for fast baseline
VAL_SUBSET_SIZE = 50_000  # Limit validation during training
FINAL_VAL_SIZE = 200_000  # Size for final metric/failure analysis (large enough to be representative)


def optimize_dataset_io(dataset):
    """
    Sorts the metadata dataframe by batch_id.
    This ensures that the dataset processes events from the same parquet batch file
    sequentially, drastically reducing I/O overhead (file opening/reading).
    """
    print(f"Optimizing I/O: Sorting {len(dataset)} events by batch_id...")
    dataset.meta_df = dataset.meta_df.sort_values("batch_id").reset_index(drop=True)
    return dataset


def main():
    # 1. Setup
    seed_everything(SEED)
    os.makedirs(WORKING_DIR, exist_ok=True)
    print(f"Running on device: {DEVICE}")

    # 2. Data Loading
    print("Initializing Datasets...")

    # Train Dataset
    train_dataset = IceCubeDataset(
        TRAIN_META_PATH, mode="train", debug_subset_size=TRAIN_SUBSET_SIZE
    )
    train_dataset = optimize_dataset_io(train_dataset)

    # Validation Dataset (for monitoring during training)
    val_dataset = IceCubeDataset(
        VAL_META_PATH, mode="val", debug_subset_size=VAL_SUBSET_SIZE
    )
    val_dataset = optimize_dataset_io(val_dataset)

    # DataLoaders
    # Note: shuffle=False is used because we sorted by batch_id for I/O speed.
    # Shuffling would negate the I/O optimization.
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Setup
    model = GeometricPulseAggregator().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = CosineDistanceLoss()

    # 4. Training Loop
    print(f"Starting training for {EPOCHS} epochs...")
    best_val_loss = float("inf")
    model_save_path = os.path.join(WORKING_DIR, "best_model.pth")

    for epoch in range(EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)

        # Validate
        val_loss, val_ang_error = validate(model, val_loader, criterion, DEVICE)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Time: {elapsed:.1f}s | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Ang Error: {val_ang_error:.4f}"
        )

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)

    print("Training complete.")

    # 5. Final Validation & Failure Analysis
    print("\nStarting Final Validation & Failure Analysis...")

    # Load best model
    model.load_state_dict(torch.load(model_save_path, map_location=DEVICE))
    model.eval()

    # Load a larger validation set for final metrics
    final_val_dataset = IceCubeDataset(
        VAL_META_PATH, mode="val", debug_subset_size=FINAL_VAL_SIZE
    )
    final_val_dataset = optimize_dataset_io(final_val_dataset)

    final_val_loader = DataLoader(
        final_val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    all_errors = []
    all_charge_sums = []
    all_aux_means = []

    total_ang_error = 0.0
    count = 0

    with torch.no_grad():
        for features, targets in final_val_loader:
            features = features.to(DEVICE)
            targets = targets.to(DEVICE)

            # Inference
            outputs = model(features)

            # --- Metric Calculation ---
            # Calculate angular error for this batch
            # Normalize
            pred_norm = torch.nn.functional.normalize(outputs, p=2, dim=1)
            target_norm = torch.nn.functional.normalize(targets, p=2, dim=1)

            # Dot product and clamp
            dot = torch.sum(pred_norm * target_norm, dim=1).clamp(-1.0, 1.0)

            # Errors in radians
            batch_errors = torch.acos(dot)

            # Accumulate for final metric
            total_ang_error += batch_errors.sum().item()
            count += features.size(0)

            # --- Failure Analysis Data Collection ---
            all_errors.append(batch_errors.cpu().numpy())

            # Feature: Signal Strength (Sum of Log-Charge)
            # features shape: (Batch, N, 6) -> [x, y, z, time, charge, aux]
            # charge is index 4
            charge_sums = features[:, :, 4].sum(dim=1)
            all_charge_sums.append(charge_sums.cpu().numpy())

            # Feature: Noise Ratio (Mean Auxiliary)
            # aux is index 5
            aux_means = features[:, :, 5].mean(dim=1)
            all_aux_means.append(aux_means.cpu().numpy())

    # Compute Final Metric
    final_metric = total_ang_error / count if count > 0 else 0.0
    print(f"Final Validation Metric: {final_metric}")

    # Compute Correlations
    flat_errors = np.concatenate(all_errors)
    flat_charges = np.concatenate(all_charge_sums)
    flat_aux = np.concatenate(all_aux_means)

    df_analysis = pd.DataFrame(
        {"error": flat_errors, "charge_sum": flat_charges, "aux_mean": flat_aux}
    )

    corr_charge = df_analysis["error"].corr(df_analysis["charge_sum"])
    corr_aux = df_analysis["error"].corr(df_analysis["aux_mean"])

    print(f"Correlation (Error vs Charge Sum): {corr_charge:.6f}")
    print(f"Correlation (Error vs Aux Mean): {corr_aux:.6f}")

    # 6. Submission
    print("\nGenerating Submission for Test Set...")

    # Prepare Test Dataset
    # We use the full test set (debug_subset_size=None)
    test_dataset = IceCubeDataset(TEST_META_PATH, mode="test", debug_subset_size=None)
    test_dataset = optimize_dataset_io(test_dataset)

    # Note: predict_test_set expects a loader. We create one with our optimized dataset.
    # We must ensure shuffle=False to match metadata order (which we sorted).
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Call the library function to generate submission
    predict_test_set(
        model_path=model_save_path,
        output_path=SUBMISSION_PATH,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        device=DEVICE,
        debug_subset_size=None,  # We already handled dataset creation, but the function re-creates it.
        # Wait, predict_test_set creates its own dataset internally.
        # We should bypass that or ensure it uses optimization.
        # Since we cannot modify library.inference, we will use the function as is.
        # However, without sorting, it will be slow.
        # BUT, predict_test_set takes a loader? No, looking at library.inference:
        # It creates the dataset internally: test_dataset = IceCubeDataset(...)
        # This means we cannot inject our optimized dataset into predict_test_set easily
        # without modifying the library.
        #
        # RE-READING library.inference.py:
        # It has a function `predict_test_set` that creates the dataset.
        # It DOES NOT take a loader as argument.
        # However, `library.model.predict_and_submit` DOES take a loader.
        # `library.train.run_training` calls `predict_and_submit`.
        # I should use `library.model.predict_and_submit` directly with my optimized loader!
    )

    # Correct approach: Use predict_and_submit from library.model with our optimized loader
    from library.model import predict_and_submit

    predict_and_submit(model, test_loader, DEVICE, SUBMISSION_PATH)


if __name__ == "__main__":
    main()
