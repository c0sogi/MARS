import sys
import os
import torch
import numpy as np
import warnings

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

from library.config import Config
from library.utils import seed_everything, calculate_rmse
from library.train import run_training_session
from library.inference import load_trained_models, predict_with_tta, generate_submission
from library.dataset import get_dataloaders


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration Overrides for Fast Baseline
    # -------------------------------------------------------------------------
    # The task requires a fast baseline execution.
    # We limit the number of epochs to ensure the script completes quickly.
    # The full solution uses 1000, but for this verification run, we use 50.
    Config.NUM_EPOCHS = 50

    # The dataset is small (92 training samples), so we use the full set.
    # Limiting it further would likely prevent any meaningful learning.
    Config.MAX_TRAIN_SAMPLES = None

    # Ensure reproducibility
    seed_everything(Config.SEED)

    # -------------------------------------------------------------------------
    # 2. Training Phase
    # -------------------------------------------------------------------------
    # Iterate through the defined streams (Context and Texture) and train their ensembles.
    trained_models = []

    for stream in Config.STREAMS:
        for seed in stream["seeds"]:
            # Train the model. load_cached_data=True utilizes ./working/*.npz if present.
            # This function saves the best model to disk and returns its path.
            model_path = run_training_session(stream, seed, load_cached_data=True)
            trained_models.append(model_path)

    # -------------------------------------------------------------------------
    # 3. Validation & Failure Analysis
    # -------------------------------------------------------------------------
    device = torch.device(Config.DEVICE)

    # Load all trained models for the heterogeneous ensemble
    models = load_trained_models(device)

    if not models:
        print("Error: No models loaded. Cannot proceed with validation.")
        return

    # Load Validation Data
    # We use Stream A's configuration to determine the patch size for validation loading.
    # Note: The library's validation loader applies a deterministic CenterCrop.
    _, val_loader = get_dataloaders(Config.STREAM_A, load_cached_data=True)

    all_preds = []
    all_targets = []

    # Statistics for Failure Analysis
    input_means = []
    input_stds = []
    sample_errors = []

    # Set models to evaluation mode
    for model in models:
        model.eval()

    with torch.no_grad():
        for inputs, targets, _ in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # --- Ensemble Inference ---
            ensemble_pred = None

            for model in models:
                # Apply Test-Time Augmentation (TTA) for each model
                tta_pred = predict_with_tta(model, inputs)

                if ensemble_pred is None:
                    ensemble_pred = tta_pred
                else:
                    ensemble_pred += tta_pred

            # Average predictions across the ensemble
            ensemble_pred /= len(models)

            # Clamp values to valid intensity range [0, 1]
            ensemble_pred = torch.clamp(ensemble_pred, 0, 1)

            # Store for global metric calculation
            all_preds.append(ensemble_pred.cpu())
            all_targets.append(targets.cpu())

            # --- Data Collection for Failure Analysis ---
            # Calculate error and stats per image in the batch
            batch_size = inputs.size(0)
            for i in range(batch_size):
                # Input Image Statistics (Mean & Std)
                img_np = inputs[i, 0].cpu().numpy()
                input_means.append(np.mean(img_np))
                input_stds.append(np.std(img_np))

                # Prediction Error (RMSE for this specific image)
                tgt_np = targets[i, 0].cpu().numpy()
                pred_np = ensemble_pred[i, 0].cpu().numpy()
                error = np.sqrt(np.mean((tgt_np - pred_np) ** 2))
                sample_errors.append(error)

    # Calculate Global Validation Metric
    y_pred_all = torch.cat(all_preds)
    y_true_all = torch.cat(all_targets)

    final_rmse = calculate_rmse(y_true_all, y_pred_all)

    # Print the required metric
    print(f"Final Validation Metric: {final_rmse}")

    # --- Failure Analysis Report ---
    sample_errors = np.array(sample_errors)
    input_means = np.array(input_means)
    input_stds = np.array(input_stds)

    # Calculate Pearson Correlation
    # np.corrcoef returns the correlation matrix
    if len(sample_errors) > 1:
        corr_mean = np.corrcoef(sample_errors, input_means)[0, 1]
        corr_std = np.corrcoef(sample_errors, input_stds)[0, 1]
    else:
        corr_mean = 0.0
        corr_std = 0.0

    print(f"Correlation (Error vs Input Mean): {corr_mean}")
    print(f"Correlation (Error vs Input Std): {corr_std}")

    # -------------------------------------------------------------------------
    # 4. Submission Generation
    # -------------------------------------------------------------------------
    # Generate submission only if the validation metric is below the specified threshold.
    THRESHOLD = 0.011870221132053216

    if final_rmse < THRESHOLD:
        generate_submission()
    else:
        # Submission skipped as metric did not meet threshold
        pass


if __name__ == "__main__":
    main()
