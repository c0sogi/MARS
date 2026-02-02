import sys
import os
import numpy as np
import torch
import pandas as pd
from scipy.stats import pearsonr

# Import library components
from library.config import Config
from library.utils import seed_everything, calculate_rmse, apply_tta, reverse_tta
from library.train_engine import run_training
from library.inference_engine import load_ensemble, create_submission_file
from library.dataset import get_dataloaders


def run():
    # ==========================================
    # 1. Configuration Override for Fast Baseline
    # ==========================================
    # Adjust parameters to ensure the script completes within the 2-hour limit
    # while utilizing the A100 GPU efficiently.
    print("Configuring parameters for fast baseline execution...")
    Config.NUM_EPOCHS = 10  # Reduced from 60 to 10 for speed
    Config.ENSEMBLE_SIZE = 3  # Reduced from 5 to 3
    Config.BATCH_SIZE = (
        512  # Increased to 512 to utilize A100 memory and speed up training
    )

    # Ensure working directory exists (Config creates it on import, but good practice)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # ==========================================
    # 2. Ensemble Training
    # ==========================================
    print(f"Starting training of {Config.ENSEMBLE_SIZE} ensemble models...")
    ensemble_rmses = []

    for i in range(Config.ENSEMBLE_SIZE):
        model_name = f"model_{i}"
        # Offset seed for each model to ensure ensemble diversity
        current_seed = Config.SEED + i

        print(
            f"\n--- Training Model {i+1}/{Config.ENSEMBLE_SIZE} (Seed: {current_seed}) ---"
        )
        best_rmse = run_training(
            model_name=model_name,
            seed=current_seed,
            epochs=Config.NUM_EPOCHS,
            load_cached_data=True,
            debug=False,
        )
        ensemble_rmses.append(best_rmse)

    print(f"\nEnsemble training complete. Individual Best RMSEs: {ensemble_rmses}")

    # ==========================================
    # 3. Validation & Failure Analysis
    # ==========================================
    print("\nStarting Validation and Failure Analysis...")
    device = torch.device(Config.DEVICE)

    # Load the trained ensemble
    models = load_ensemble(device, ensemble_size=Config.ENSEMBLE_SIZE)

    # Get Validation DataLoader (we only need val_loader here)
    # load_cached_data=True ensures we use the data processed during training
    _, val_loader = get_dataloaders(load_cached_data=True)

    all_preds = []
    all_targets = []
    all_inputs = []

    print("Running inference on validation set with TTA...")

    # Validation Inference Loop
    # We manually implement the ensemble + TTA logic here to compute the metric
    # on the validation set efficiently.
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_loader):
            inputs = inputs.to(device)
            # targets are needed for metric calculation

            # Accumulate ensemble predictions
            # The model predicts the NOISE residual.
            # We average the noise predictions from all models and TTA steps.
            batch_noise_pred = torch.zeros_like(inputs)

            total_passes = len(models) * Config.TTA_STEPS

            for model in models:
                for k in range(Config.TTA_STEPS):
                    # Apply TTA
                    aug_input = apply_tta(inputs, k)

                    # Predict Noise
                    aug_noise = model(aug_input)

                    # Reverse TTA
                    pred_noise = reverse_tta(aug_noise, k)

                    batch_noise_pred += pred_noise

            # Average
            batch_noise_pred /= total_passes

            # Reconstruct Clean Image: Clean = Noisy - Predicted Noise
            clean_pred = inputs - batch_noise_pred

            # Clamp to valid range [0, 1]
            clean_pred = torch.clamp(clean_pred, 0.0, 1.0)

            # Store results on CPU to save GPU memory
            all_preds.append(clean_pred.cpu())
            all_targets.append(targets.cpu())
            all_inputs.append(inputs.cpu())

    # Concatenate all batches
    y_pred = torch.cat(all_preds)
    y_true = torch.cat(all_targets)
    inputs_flat = torch.cat(all_inputs)

    # Calculate Final Validation Metric
    val_rmse = calculate_rmse(y_true, y_pred)
    print(f"Final Validation Metric: {val_rmse:.18f}")

    # --- Failure Analysis ---
    print("Performing Failure Analysis...")

    # Calculate absolute errors
    # Ensure we are working with numpy arrays for stats
    y_pred_np = y_pred.numpy()
    y_true_np = y_true.numpy()
    inputs_np = inputs_flat.numpy()

    errors = np.abs(y_pred_np - y_true_np)

    # Flatten arrays
    flat_errors = errors.flatten()
    flat_inputs = inputs_np.flatten()

    # Calculate Pearson Correlation
    # We sample if the dataset is excessively large to keep analysis fast,
    # but with ~100MB of data, full calculation is feasible.
    # 23 images * ~1500 patches * 2500 pixels ~ 86M pixels.
    # Scipy can handle this, but let's be safe with memory.
    if len(flat_errors) > 10_000_000:
        # Sample 10M points for correlation
        indices = np.random.choice(len(flat_errors), 10_000_000, replace=False)
        corr, _ = pearsonr(flat_inputs[indices], flat_errors[indices])
    else:
        corr, _ = pearsonr(flat_inputs, flat_errors)

    print(f"Correlation (Input Intensity vs Error Magnitude): {corr:.6f}")
    if abs(corr) > 0.3:
        print(
            "Observation: Significant correlation found. Error is dependent on pixel intensity."
        )
    else:
        print(
            "Observation: Low correlation. Error is relatively independent of pixel intensity."
        )

    # ==========================================
    # 4. Submission
    # ==========================================
    THRESHOLD = 0.011577641381826402

    if val_rmse < THRESHOLD:
        print(
            f"\nValidation RMSE ({val_rmse:.6f}) is better than threshold ({THRESHOLD})."
        )
        print("Generating submission file...")
        create_submission_file()
    else:
        print(
            f"\nValidation RMSE ({val_rmse:.6f}) did not meet threshold ({THRESHOLD})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    run()
