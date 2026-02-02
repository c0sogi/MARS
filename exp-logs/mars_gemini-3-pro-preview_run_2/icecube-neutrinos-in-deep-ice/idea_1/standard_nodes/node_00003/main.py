import os
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Import from provided library files
from library.config import (
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    SUBMISSION_DIR,
    DEVICE,
    SEED,
    BATCH_SIZE,
    NUM_WORKERS,
    CACHE_DIR,
)
from library.utils import seed_everything, vector_to_angles
from library.dataset import IceCubeDataset
from library.model import NeutrinoBiGRU
from library.loss import CosineDistanceLoss
from library.engine import train_with_early_stopping


def calculate_elementwise_angular_error(pred_vectors, target_angles):
    """
    Calculates angular error for each element in the batch.
    """
    # Normalize predicted vectors
    pred_norm = F.normalize(pred_vectors, p=2, dim=1)

    # Convert targets to vectors
    azimuth = target_angles[:, 0]
    zenith = target_angles[:, 1]

    sin_zenith = torch.sin(zenith)
    true_x = torch.cos(azimuth) * sin_zenith
    true_y = torch.sin(azimuth) * sin_zenith
    true_z = torch.cos(zenith)

    true_vector = torch.stack([true_x, true_y, true_z], dim=1)

    # Cosine similarity
    cosine_sim = torch.sum(pred_norm * true_vector, dim=1)
    cosine_sim = torch.clamp(cosine_sim, -1.0, 1.0)

    # Angular error in radians
    errors = torch.acos(cosine_sim)
    return errors


def main():
    # 1. Setup
    seed_everything(SEED)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    print(f"Running on device: {DEVICE}")

    # 2. Load and Prepare Data
    print("Loading metadata...")
    train_meta = pd.read_parquet(TRAIN_META_PATH)
    val_meta = pd.read_parquet(VAL_META_PATH)

    # Sampling for Fast Baseline
    # We use a subset to ensure the code completes quickly within the limit
    N_TRAIN = 200000
    N_VAL = 50000

    print(f"Sampling {N_TRAIN} training events and {N_VAL} validation events...")
    train_meta = train_meta.sample(n=N_TRAIN, random_state=SEED).reset_index(drop=True)

    # Sort validation by batch_id to optimize IO (sequential file reading)
    val_meta = (
        val_meta.sample(n=N_VAL, random_state=SEED)
        .sort_values("batch_id")
        .reset_index(drop=True)
    )

    # Initialize Datasets
    # Cite debug_lesson_1: Account for Decompression Expansion.
    # 170MB Parquet -> ~1.5GB DataFrame.
    # Old limit: 40 * 1.5GB * 4 workers = 240GB (OOM).
    # New limit: 5 * 1.5GB * 4 workers = 30GB (Safe).
    print("Initializing Datasets...")
    train_dataset = IceCubeDataset(train_meta, mode="train", cache_limit=5)
    val_dataset = IceCubeDataset(val_meta, mode="train", cache_limit=5)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
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
    model = NeutrinoBiGRU().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = CosineDistanceLoss()

    # 4. Training
    print("Starting training...")
    model, history = train_with_early_stopping(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        num_epochs=5,  # Limited epochs for fast baseline
        patience=2,
        device=DEVICE,
        save_path=os.path.join(CACHE_DIR, "best_model.pth"),
    )

    # 5. Validation & Failure Analysis
    print("Performing full validation and failure analysis...")
    model.eval()

    all_errors = []
    all_n_pulses = []
    all_total_charge = []

    running_mae_sum = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            # Forward
            outputs = model(inputs)

            # Calculate Errors
            batch_errors = calculate_elementwise_angular_error(outputs, targets)
            batch_errors_np = batch_errors.cpu().numpy()

            all_errors.extend(batch_errors_np)
            running_mae_sum += np.sum(batch_errors_np)
            total_samples += inputs.size(0)

            # Extract Features for Analysis
            # inputs: (B, L, 6) -> [x, y, z, t, c, a]
            # c is log1p(charge) at index 4
            c_tensor = inputs[:, :, 4]

            # Revert log1p to get approximate charge
            charge_tensor = torch.expm1(c_tensor)

            # Calculate total charge per event
            batch_charge = torch.sum(charge_tensor, dim=1).cpu().numpy()

            # Calculate n_pulses (count where charge > 0, assuming padding is 0)
            # Since data is normalized/transformed, exact 0 check might be risky if not careful,
            # but padding in dataset.py is explicitly np.zeros.
            # expm1(0) = 0. So checking > 1e-6 is safe.
            batch_pulses = torch.sum(charge_tensor > 1e-6, dim=1).cpu().numpy()

            all_total_charge.extend(batch_charge)
            all_n_pulses.extend(batch_pulses)

    final_metric = running_mae_sum / total_samples
    print(f"Final Validation Metric: {final_metric:.6f}")

    # Failure Analysis
    all_errors = np.array(all_errors)
    all_n_pulses = np.array(all_n_pulses)
    all_total_charge = np.array(all_total_charge)

    # Correlations
    corr_pulses, _ = pearsonr(all_errors, all_n_pulses)
    corr_charge, _ = pearsonr(all_errors, all_total_charge)

    print("\nFailure Analysis:")
    print(f"Correlation (Error vs n_pulses): {corr_pulses:.4f}")
    print(f"Correlation (Error vs total_charge): {corr_charge:.4f}")

    # 6. Submission Generation
    print("\nGenerating submission for test set...")

    # Load Test Metadata
    test_meta = pd.read_parquet(TEST_META_PATH)
    # Sort by batch_id for efficient sequential reading
    test_meta = test_meta.sort_values("batch_id").reset_index(drop=True)

    test_dataset = IceCubeDataset(test_meta, mode="test", cache_limit=5)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE * 2,  # Larger batch for inference
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    pred_azimuths = []
    pred_zeniths = []

    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(DEVICE)

            outputs = model(inputs)

            # Normalize
            pred_vectors = F.normalize(outputs, p=2, dim=1).cpu().numpy()

            # Convert to angles
            az, zen = vector_to_angles(
                pred_vectors[:, 0], pred_vectors[:, 1], pred_vectors[:, 2]
            )

            pred_azimuths.extend(az)
            pred_zeniths.extend(zen)

    # Create Submission DataFrame
    submission_df = pd.DataFrame(
        {
            "event_id": test_meta["event_id"],
            "azimuth": pred_azimuths,
            "zenith": pred_zeniths,
        }
    )

    # Save
    save_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")
    print(f"Submission shape: {submission_df.shape}")


if __name__ == "__main__":
    main()
