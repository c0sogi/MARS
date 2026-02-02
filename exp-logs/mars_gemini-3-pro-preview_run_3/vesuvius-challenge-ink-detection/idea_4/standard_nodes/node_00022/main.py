import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from pathlib import Path

# Import from provided libraries
from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders, InkDataset
from library.model import HDNPCA
from library.engine import train_one_epoch, evaluate
from library.inference import predict_full_map, find_best_threshold, generate_submission


def run_failure_analysis(prob_maps, dataset):
    """
    Calculates the correlation between model error and input pixel intensity.
    """
    print("\n--- Failure Analysis ---")

    errors = []
    intensities = []

    for frag in dataset.fragments:
        fid = frag["id"]
        if frag["label"] is None:
            continue

        # Get Data
        pred_prob = prob_maps[fid]
        target = frag["label"]
        mask = frag["mask"]
        volume = frag["volume"]  # Shape (D, H, W)

        # Calculate Mean Intensity Image (Input Feature)
        # Volume is already normalized, but relative intensity still holds
        mean_intensity = np.mean(volume, axis=0)

        # Calculate Error Magnitude
        error_map = np.abs(pred_prob - target)

        # Select only valid pixels
        valid_indices = mask > 0

        errors.extend(error_map[valid_indices])
        intensities.extend(mean_intensity[valid_indices])

    # Convert to arrays
    errors = np.array(errors, dtype=np.float32)
    intensities = np.array(intensities, dtype=np.float32)

    # Calculate Correlation
    if len(errors) > 0:
        corr, _ = pearsonr(errors, intensities)
        print(f"Correlation between Error Magnitude and Input Intensity: {corr:.4f}")
    else:
        print("No valid validation pixels found for analysis.")


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")

    # Ensure working directory exists
    Config.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Data Loading
    # Using load_cached=True to speed up if data exists
    # Limiting training samples per epoch to 2000 for fast baseline
    print("Initializing DataLoaders...")
    train_loader, val_loader = get_dataloaders(load_cached=True, debug=Config.DEBUG)

    # 3. Model Initialization
    model = HDNPCA().to(device)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # 4. Training Loop
    best_score = -1.0
    patience_counter = 0

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{Config.EPOCHS}")

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        # Validate
        # Note: evaluate() uses a default threshold of 0.5 for quick checking during training
        val_loss, val_f05 = evaluate(model, val_loader, device, threshold=0.5)

        # Save Best Model
        if val_f05 > best_score:
            best_score = val_f05
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"New best model saved with F0.5: {best_score:.6f}")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    # 5. Final Validation & Threshold Tuning
    print("\nLoading best model for final validation...")
    if Config.BEST_MODEL_PATH.exists():
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))

    # We need the dataset object to access fragments for full map reconstruction
    # Re-instantiate val dataset to ensure clean state or use the one from loader
    val_ds = val_loader.dataset

    print("Generating full validation probability maps...")
    val_probs = predict_full_map(model, val_loader, val_ds, device, use_tta=True)

    # Find optimal threshold
    best_threshold = find_best_threshold(val_probs, val_ds)

    # Calculate Final Metric with Best Threshold
    # We re-calculate explicitly to print in the required format
    tp_sum = 0
    fp_sum = 0
    fn_sum = 0
    smooth = 1e-6
    beta = 0.5

    for frag in val_ds.fragments:
        fid = frag["id"]
        if frag["label"] is None:
            continue

        prob = val_probs[fid]
        pred = (prob > best_threshold).astype(np.uint8)
        target = frag["label"]
        mask = frag["mask"]

        pred = pred * mask
        target = target * mask

        tp = np.sum((pred == 1) & (target == 1))
        fp = np.sum((pred == 1) & (target == 0))
        fn = np.sum((pred == 0) & (target == 1))

        tp_sum += tp
        fp_sum += fp
        fn_sum += fn

    precision = tp_sum / (tp_sum + fp_sum + smooth)
    recall = tp_sum / (tp_sum + fn_sum + smooth)
    beta_sq = beta**2
    final_metric = (
        (1 + beta_sq) * (precision * recall) / ((beta_sq * precision) + recall + smooth)
    )

    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    run_failure_analysis(val_probs, val_ds)

    # Clean up memory
    del val_probs, val_loader, val_ds, train_loader
    torch.cuda.empty_cache()

    # 7. Submission
    TARGET_METRIC = 0.39266693592071533

    if final_metric > TARGET_METRIC:
        print("\nValidation metric meets criteria. Generating submission...")

        # Load Test Data
        # We define a custom loader function call or use InkDataset directly
        test_ds = InkDataset("test", load_cached=True)
        test_loader = torch.utils.data.DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Predict
        test_probs = predict_full_map(model, test_loader, test_ds, device, use_tta=True)

        # Generate CSV
        generate_submission(test_probs, test_ds, best_threshold, Config.SUBMISSION_PATH)

    else:
        print(
            f"\nValidation metric {final_metric} does not exceed {TARGET_METRIC}. Skipping submission."
        )


if __name__ == "__main__":
    main()
