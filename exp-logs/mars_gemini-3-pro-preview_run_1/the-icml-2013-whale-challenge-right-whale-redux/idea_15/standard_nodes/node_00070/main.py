import os
import numpy as np
import pandas as pd
import torch
import soundfile as sf
import warnings

# Import provided library modules
from library.config import TrainConfig, PathConfig
from library.utils import set_seed, calculate_auc
from library.dataset import get_dataloaders
from library.model import SpecFPN_CRNN
from library.trainer import fit_model, generate_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run():
    """
    Main orchestration function for the Right Whale Detection pipeline.
    """
    # 1. Setup and Data Loading
    print("Initializing pipeline...")
    # Set the first seed for global operations
    set_seed(TrainConfig.SEEDS[0])

    # Load DataLoaders (utilizing cached .npy files for speed)
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 2. Train Ensemble
    # We train multiple models with different seeds to boost performance and stability
    trained_models = []

    print(f"\nStarting Ensemble Training with {len(TrainConfig.SEEDS)} models...")

    for seed in TrainConfig.SEEDS:
        print(f"\n--- Training Model with Seed {seed} ---")

        # Initialize the architecture
        model = SpecFPN_CRNN()

        # Train the model
        # fit_model handles optimization, scheduling, and saving the best checkpoint
        model = fit_model(model, train_loader, val_loader, seed)

        # Add to ensemble
        trained_models.append(model)

    # 3. Ensemble Validation
    print("\n--- Running Ensemble Validation ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Collect Ground Truth Targets
    # val_loader is sequential (shuffle=False), so we can iterate once to get targets
    y_true = []
    for _, targets in val_loader:
        y_true.append(targets.numpy())
    y_true = np.concatenate(y_true)

    # Collect Predictions from all models
    ensemble_preds = []

    for i, model in enumerate(trained_models):
        model.eval()
        model.to(device)

        preds = []
        with torch.no_grad():
            for inputs, _ in val_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                preds.append(outputs.cpu().numpy().flatten())

        ensemble_preds.append(np.concatenate(preds))

    # Average predictions (Ensemble)
    y_pred_avg = np.mean(ensemble_preds, axis=0)

    # Calculate Final Metric
    final_metric = calculate_auc(y_true, y_pred_avg)
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\n--- Performing Failure Analysis ---")

    # Calculate Error Magnitude
    errors = np.abs(y_true - y_pred_avg)

    # Extract Features for Correlation Analysis (Audio Duration)
    # We read the original file headers to get the exact duration
    val_df = pd.read_csv(PathConfig.VAL_CSV)
    durations = []

    print("Extracting validation file metadata...")
    for _, row in val_df.iterrows():
        filepath = os.path.join(PathConfig.INPUT_ROOT, row["filepath"])
        try:
            # Efficiently read only the header
            info = sf.info(filepath)
            durations.append(info.duration)
        except Exception:
            # Fallback if file is corrupt (unlikely given checks)
            durations.append(2.0)

    durations = np.array(durations)

    # Calculate Correlation
    if np.std(durations) > 0 and np.std(errors) > 0:
        corr = np.corrcoef(errors, durations)[0, 1]
        print(f"Correlation between Error and Audio Duration: {corr:.6f}")
    else:
        print(
            "Correlation between Error and Audio Duration: 0.000000 (Constant feature or error)"
        )

    # 5. Submission Generation
    threshold = 0.994932894209377

    if final_metric > threshold:
        print(f"\nMetric {final_metric} exceeds threshold {threshold}.")
        print("Generating submission file...")
        generate_submission(trained_models, test_loader)
    else:
        print(f"\nMetric {final_metric} does not exceed threshold {threshold}.")
        print("Skipping submission generation.")


if __name__ == "__main__":
    run()
