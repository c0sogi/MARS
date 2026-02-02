import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, calculate_rmse
from library.data_loader import load_dataset_patches
from library.trainer import train_ensemble_member
from library.predictor import inference_pipeline
from library.network import DnCNN


def run():
    # =========================================================================
    # 1. Configuration Overrides for Fast Baseline
    # =========================================================================
    # Adjusting parameters to ensure execution within the 1-hour limit
    # while maintaining sufficient learning capacity.
    print("Configuring Fast Baseline parameters...")

    # Reduce ensemble size for speed (3 instead of 5)
    Config.ENSEMBLE_SIZE = 3

    # Optimize Curriculum Schedule
    # Stage 1: Fast convergence on sparse data
    Config.STAGE_1_EPOCHS = 3
    Config.STRIDE_SPARSE = 40  # Larger stride = fewer patches

    # Stage 2: Refinement on dense data
    Config.STAGE_2_EPOCHS = 3
    Config.STRIDE_DENSE = 20  # Moderate density

    # Hardware Optimization
    # Reduced batch size to fit in available VRAM (~16GB detected in logs)
    Config.BATCH_SIZE = 128

    # =========================================================================
    # 2. Train Ensemble
    # =========================================================================
    seeds = Config.get_ensemble_seeds()
    for i in range(Config.ENSEMBLE_SIZE):
        print(f"\nTraining Ensemble Member {i+1}/{Config.ENSEMBLE_SIZE}")
        # Train the member; this handles saving the best checkpoint to working/
        train_ensemble_member(i, seeds[i])

    # =========================================================================
    # 3. Ensemble Validation
    # =========================================================================
    print("\n--- Starting Ensemble Validation ---")
    device = torch.device(Config.DEVICE)

    # Load validation data
    # Note: load_dataset_patches uses Config.STRIDE_SPARSE for 'val' mode
    val_patches, val_targets = load_dataset_patches("val", load_cached_data=True)

    # Create DataLoader for efficient batch inference
    # val_patches shape: (N, 1, H, W)
    val_dataset = TensorDataset(
        torch.from_numpy(val_patches).float(), torch.from_numpy(val_targets).float()
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Accumulate predictions from all ensemble members
    ensemble_accumulated_preds = np.zeros_like(val_patches)
    valid_models_count = 0

    for i in range(Config.ENSEMBLE_SIZE):
        model_path = os.path.join(Config.WORKING_DIR, f"model_{i}.pth")
        if not os.path.exists(model_path):
            print(f"Warning: Model {i} checkpoint not found. Skipping.")
            continue

        print(f"Evaluating Member {i}...")

        # Initialize model structure
        model = DnCNN(
            in_channels=Config.IN_CHANNELS,
            out_channels=Config.OUT_CHANNELS,
            num_features=Config.NUM_FEATURES,
            num_blocks=Config.NUM_RES_BLOCKS,
        ).to(device)

        # Load weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        member_preds = []
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(device)

                # Predict noise residual
                pred_noise = model(batch_x)

                # Reconstruct clean image: Clean = Noisy - Noise
                pred_clean = batch_x - pred_noise

                member_preds.append(pred_clean.cpu().numpy())

        # Concatenate batches for this member
        member_preds_np = np.concatenate(member_preds, axis=0)
        ensemble_accumulated_preds += member_preds_np
        valid_models_count += 1

    if valid_models_count == 0:
        print("Error: No valid models trained. Exiting.")
        return

    # Average predictions
    ensemble_preds = ensemble_accumulated_preds / valid_models_count

    # Clip to valid pixel range [0, 1]
    ensemble_preds = np.clip(ensemble_preds, 0.0, 1.0)

    # Calculate RMSE
    val_rmse = calculate_rmse(val_targets, ensemble_preds)

    # REQUIRED OUTPUT: Print the final validation metric
    print(f"Final Validation Metric: {val_rmse}")

    # =========================================================================
    # 4. Failure Analysis
    # =========================================================================
    print("\n--- Failure Analysis ---")

    # Calculate error magnitude per patch (MSE)
    # shape: (N, 1, H, W) -> (N,)
    errors = (ensemble_preds - val_targets) ** 2
    mse_per_patch = np.mean(errors, axis=(1, 2, 3))

    # Input features per patch
    # Mean intensity
    input_mean = np.mean(val_patches, axis=(1, 2, 3))
    # Standard deviation (contrast/texture)
    input_std = np.std(val_patches, axis=(1, 2, 3))

    # Calculate Correlations using numpy
    if len(mse_per_patch) > 1:
        corr_mean = np.corrcoef(mse_per_patch, input_mean)[0, 1]
        corr_std = np.corrcoef(mse_per_patch, input_std)[0, 1]
    else:
        corr_mean = 0.0
        corr_std = 0.0

    print(f"Correlation (Error Magnitude vs. Input Mean Intensity): {corr_mean:.4f}")
    print(f"Correlation (Error Magnitude vs. Input Std Dev): {corr_std:.4f}")

    # Interpretations
    if corr_mean > 0.2:
        print("Insight: Higher errors occur in brighter areas (background).")
    elif corr_mean < -0.2:
        print("Insight: Higher errors occur in darker areas (text/ink).")

    if corr_std > 0.2:
        print("Insight: Higher errors occur in high-contrast areas (edges/strokes).")

    # =========================================================================
    # 5. Submission
    # =========================================================================
    # Threshold defined in the task description
    THRESHOLD = 0.011577641381826402

    if val_rmse < THRESHOLD:
        print(
            f"\nValidation metric ({val_rmse:.8f}) meets threshold ({THRESHOLD:.8f})."
        )
        print("Proceeding to generate submission...")
        # Run the inference pipeline provided in the library
        inference_pipeline()
    else:
        print(
            f"\nValidation metric ({val_rmse:.8f}) does NOT meet threshold ({THRESHOLD:.8f})."
        )
        print("Submission generation skipped.")


if __name__ == "__main__":
    run()
