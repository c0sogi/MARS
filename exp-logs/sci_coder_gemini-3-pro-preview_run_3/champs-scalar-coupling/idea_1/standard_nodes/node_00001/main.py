import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings

# Import provided library modules
import library.config as config
from library.utils import seed_everything
from library.data_factory import get_dataloaders
from library.model_arch import CouplingPredictor
from library.training_engine import ModelTrainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # 1. Setup and Configuration
    # Set seeds for reproducibility across numpy, torch, etc.
    seed_everything(config.SEED)

    print("==== Starting Baseline Run ====")

    # 2. Data Loading
    # We use debug mode with a subset of data to ensure the baseline runs quickly (within the time limit).
    # 50,000 samples provide a good balance between speed and representation for a baseline.
    print("Initializing DataLoaders (Debug Mode)...")
    train_loader, val_loader, test_loader = get_dataloaders(
        train_path=config.TRAIN_META_PATH,
        val_path=config.VAL_META_PATH,
        test_path=config.TEST_META_PATH,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        debug=True,
        debug_size=50000,
    )

    # 3. Model Initialization
    print("Initializing Model...")
    model = CouplingPredictor()

    # 4. Training
    print("Initializing Trainer...")
    trainer = ModelTrainer(model, train_loader, val_loader, test_loader)

    print("Starting Training Loop...")
    # The trainer handles the training loop, validation per epoch, and early stopping.
    trainer.run()

    # 5. Validation Assessment
    print("\n==== Validation Assessment ====")
    # Load the best model state saved during training for analysis
    best_model_path = os.path.join(config.WORKING_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=config.DEVICE))
    else:
        print("Warning: Best model not found. Using current model state.")

    # Calculate final metric on the validation set
    val_metric = trainer.validate()
    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_metric}")

    # 6. Failure Analysis
    print("\n==== Failure Analysis ====")
    analyze_failures(model, val_loader, trainer.scaler)

    # 7. Submission Generation
    print("\n==== Generating Submission ====")
    # Generates predictions on the test set and saves to ./submission/submission.csv
    trainer.predict()
    print("Run complete.")


def analyze_failures(model, val_loader, scaler):
    """
    Performs failure analysis on the validation set.
    Calculates correlations between error and input features to identify systematic issues.
    """
    model.eval()
    device = config.DEVICE

    all_errors = []
    all_dists = []
    all_targets = []
    all_types = []

    print("Computing errors on validation set for analysis...")
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)

            # Forward pass
            preds_scaled = model(batch)

            # Inverse transform to get real values for error calculation
            batch_types = batch.type_idx.view(-1)
            preds = scaler.inverse_transform(preds_scaled, batch_types)
            targets = batch.y.view(-1)

            # Calculate Absolute Error
            abs_error = torch.abs(preds - targets)

            # Store data for analysis
            all_errors.append(abs_error.cpu().numpy())
            all_dists.append(batch.dist.view(-1).cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_types.append(batch_types.cpu().numpy())

    # Concatenate results from all batches
    errors = np.concatenate(all_errors)
    dists = np.concatenate(all_dists)
    targets = np.concatenate(all_targets)
    types = np.concatenate(all_types)

    # 1. Correlation Analysis
    # We use numpy to calculate Pearson correlation
    # Check correlation between Error and Distance
    corr_dist = np.corrcoef(errors, dists)[0, 1]

    # Check correlation between Error and Target Magnitude (absolute value of target)
    # This tells us if the model struggles more with larger coupling constants
    corr_target = np.corrcoef(errors, np.abs(targets))[0, 1]

    print("Correlation between Absolute Error and Input Features:")
    print(f"  Distance: {corr_dist:.4f}")
    print(f"  Target Magnitude (abs): {corr_target:.4f}")

    # 2. Error by Coupling Type
    print("\nMean Absolute Error by Coupling Type:")
    df_analysis = pd.DataFrame({"error": errors, "type": types})
    type_errors = df_analysis.groupby("type")["error"].mean()

    # Map integer types back to string names for readability
    inv_type_map = {v: k for k, v in config.TYPE_MAP.items()}

    for type_idx, mean_err in type_errors.items():
        type_name = inv_type_map.get(type_idx, f"Type {type_idx}")
        print(f"  {type_name}: {mean_err:.4f}")


if __name__ == "__main__":
    main()
