import os
import sys
import numpy as np
import torch
from scipy.stats import pearsonr

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, calculate_rmse
from library.train import train_model
from library.inference import predict_and_save
from library.model import ResDnCNN
from library.data_loader import get_dataloaders


def run():
    # 1. Setup and Config Overrides for Fast Baseline
    set_seed(Config.SEED)

    # Adjust hyperparameters to ensure execution within time limits and limit training samples
    # Increasing stride reduces the number of patches, speeding up extraction and training
    Config.STRIDE = 25
    # Reduce epochs for a quick baseline run
    Config.NUM_EPOCHS = 10

    print(f"Running with Stride={Config.STRIDE}, Epochs={Config.NUM_EPOCHS}")

    # 2. Train the Model
    # This will handle data extraction (if not cached) and the training loop
    # The best model will be saved to Config.MODEL_SAVE_PATH
    train_model()

    # 3. Validation & Failure Analysis
    print("\n--- Starting Evaluation & Failure Analysis ---")

    device = torch.device(Config.DEVICE)

    # Load the best model saved during training
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"Error: Model checkpoint not found at {Config.MODEL_SAVE_PATH}")
        return

    model = ResDnCNN().to(device)
    state_dict = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Get Validation Data
    # load_cached_data=True will use the data generated during train_model's setup
    _, val_loader = get_dataloaders(load_cached_data=True)

    all_preds = []
    all_targets = []
    all_inputs = []

    # Run Inference on Validation Set
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Predict noise residual
            outputs = model(inputs)

            # Move to CPU for analysis
            all_inputs.append(inputs.cpu().numpy())
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Concatenate all batches
    all_inputs = np.concatenate(all_inputs)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Calculate Final Validation Metric (RMSE)
    # Target is residual (Noisy - Clean). Pred is predicted residual.
    # RMSE(Pred, Target) is mathematically equivalent to RMSE(Restored, Clean)
    final_rmse = calculate_rmse(all_targets, all_preds)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_rmse}")

    # Failure Analysis
    # Calculate correlation between error magnitude and input features (pixel intensity)
    flat_inputs = all_inputs.flatten()
    flat_targets = all_targets.flatten()
    flat_preds = all_preds.flatten()

    # Error magnitude
    errors = np.abs(flat_targets - flat_preds)

    # Correlation with Input Intensity
    # If noise is signal-dependent, this might be high.
    corr_input, _ = pearsonr(errors, flat_inputs)
    print(f"Correlation (Error vs Input Intensity): {corr_input}")

    # Correlation with Noise Magnitude (Target)
    # Does the model fail more on high noise pixels?
    corr_noise, _ = pearsonr(errors, np.abs(flat_targets))
    print(f"Correlation (Error vs Noise Magnitude): {corr_noise}")

    # 4. Conditional Submission
    # Threshold from requirements
    THRESHOLD = 0.011577641381826402

    if final_rmse < THRESHOLD:
        print(f"\nMetric {final_rmse} < {THRESHOLD}. Proceeding to submission...")
        # predict_and_save handles loading the model internally and generating the CSV
        predict_and_save()
    else:
        print(f"\nMetric {final_rmse} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    run()
