import os
import random
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader

# Import provided library components
from library.config import Config
from library.data_utils import prepare_datasets
from library.train_utils import train_model, predict_and_submit
from library.model import VentilatorModel


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior where possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def perform_failure_analysis(model, val_loader, device, config):
    """
    Analyzes model failure modes on the validation set.
    Calculates correlation between error magnitude and input features
    during the inspiratory phase.
    """
    model.eval()

    all_errors = []
    all_features = []

    u_out_idx = config.INPUT_FEATURES.index("u_out")
    feature_names = config.INPUT_FEATURES

    print("\nPerforming failure analysis on validation set...")

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            final_pred, _ = model(inputs)
            final_pred = final_pred.squeeze(-1)

            # Calculate Absolute Error
            abs_error = torch.abs(final_pred - targets)

            # Mask for inspiratory phase (u_out == 0)
            u_out = inputs[:, :, u_out_idx]
            mask = u_out == 0

            # Filter data to keep only inspiratory phase
            # We flatten the batch and sequence dimensions
            valid_errors = abs_error[mask].cpu().numpy()
            valid_features = inputs[mask].cpu().numpy()

            all_errors.append(valid_errors)
            all_features.append(valid_features)

    # Concatenate all batches
    all_errors = np.concatenate(all_errors)
    all_features = np.concatenate(all_features)

    # Create DataFrame for correlation analysis
    df_analysis = pd.DataFrame(all_features, columns=feature_names)
    df_analysis["error_magnitude"] = all_errors

    # Calculate correlation with error magnitude
    correlations = df_analysis.corr()["error_magnitude"].drop("error_magnitude")

    # Sort by absolute correlation
    sorted_corr = correlations.abs().sort_values(ascending=False)

    print("Top 5 Features correlated with Error Magnitude:")
    for feat in sorted_corr.head(5).index:
        print(f"  {feat}: {correlations[feat]:.4f}")


def main():
    # 1. Setup
    set_seed(Config.SEED)

    # Configure for fast baseline execution
    # Reducing epochs to ensure completion within time limits while allowing convergence
    Config.EPOCHS = 15

    print(
        "Initializing High-Capacity Unnormalized Physics-Injected Composite CNN-LSTM-FFN Pipeline..."
    )
    Config.print_config()

    # 2. Data Preparation
    # load_cached_data=True allows using pre-engineered features from ./working if available
    train_ds, val_ds, test_ds = prepare_datasets(Config, load_cached_data=True)

    # Create DataLoaders
    # Pin memory for faster host-to-device transfer
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Training
    # train_model handles the loop, validation, and saving the best model
    best_mae = train_model(train_loader, val_loader, Config)

    # 4. Validation Metric Reporting
    # Strictly formatted output required
    print(f"Final Validation Metric: {best_mae}")

    # 5. Failure Analysis
    # Load the best model for analysis
    device = torch.device(Config.DEVICE)
    model = VentilatorModel(Config).to(device)

    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
        perform_failure_analysis(model, val_loader, device, Config)
    else:
        print("Warning: Model checkpoint not found. Skipping failure analysis.")

    # 6. Submission Generation
    # Conditional submission based on metric threshold
    THRESHOLD = 0.2164510190486908

    if best_mae < THRESHOLD:
        print(
            f"\nValidation metric ({best_mae}) meets threshold ({THRESHOLD}). Generating submission..."
        )
        predict_and_submit(test_loader, Config)
    else:
        print(
            f"\nValidation metric ({best_mae}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
