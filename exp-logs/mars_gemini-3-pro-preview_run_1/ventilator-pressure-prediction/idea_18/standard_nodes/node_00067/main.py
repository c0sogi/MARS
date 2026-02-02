import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

# Import library modules
from library.config import Config
from library.dataset import prepare_data
from library.model import VentilatorModel
from library.trainer import Trainer
from library.inference import generate_predictions


def main():
    # 1. Configuration
    # Initialize config
    config = Config(debug=False)

    # Optimization for Full Training
    # Using full dataset and extended epochs to maximize performance
    # config.EPOCHS is set in config.py (40)

    print(
        f"Configuration: Epochs={config.EPOCHS}, Dataset=Full, Device={config.DEVICE}"
    )

    # 2. Data Preparation
    print("\n=== Data Preparation ===")
    # Load full training data (cached if available)
    train_dataset = prepare_data(config, split="train", load_cached_data=True)
    val_dataset = prepare_data(config, split="val", load_cached_data=True)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DEVICE == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DEVICE == "cuda"),
    )

    # 3. Model Initialization & Training
    print("\n=== Model Training ===")
    model = VentilatorModel(config)
    trainer = Trainer(config, model)

    # Fit the model
    trainer.fit(train_loader, val_loader)

    # 4. Validation Assessment
    print("\n=== Validation Assessment ===")
    # Note: Trainer automatically loads the best model state after fit()
    val_mae = trainer.validate(val_loader)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_mae}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Analyze errors on validation set
    model.eval()
    device = torch.device(config.DEVICE)

    all_errors = []
    all_features = []

    # Get feature names for correlation analysis
    # Note: The model input is a concatenation of CONT_FEATURES and BINARY_FEATURES
    feature_names = config.CONT_FEATURES + config.BINARY_FEATURES

    print("Computing error correlations...")
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["input"].to(device)
            u_out = batch["u_out"].to(device)
            targets = batch["target"].to(device)

            # Forward pass
            final_pred, _ = model(inputs, u_out=u_out)

            # Ensure shapes are consistent (Batch, 80)
            if final_pred.dim() == 3:
                final_pred = final_pred.squeeze(-1)
            if targets.dim() == 3:
                targets = targets.squeeze(-1)
            if u_out.dim() == 3:
                u_out = u_out.squeeze(-1)

            # Calculate Absolute Error
            abs_error = torch.abs(final_pred - targets)

            # Mask: Only analyze inspiratory phase (u_out == 0) for failure analysis
            mask = u_out < 0.5

            # Flatten everything to analyze per-timestep correlation
            mask_flat = mask.flatten()

            # Filter by mask
            if mask_flat.sum() > 0:
                error_flat = abs_error.flatten()[mask_flat]

                # Inputs: (B, 80, n_features) -> (B*80, n_features)
                inputs_flat = inputs.reshape(-1, inputs.shape[-1])[mask_flat]

                all_errors.append(error_flat.cpu().numpy())
                all_features.append(inputs_flat.cpu().numpy())

    if all_errors:
        all_errors = np.concatenate(all_errors)
        all_features = np.concatenate(all_features)

        # Create DataFrame
        df_analysis = pd.DataFrame(all_features, columns=feature_names)
        df_analysis["error_magnitude"] = all_errors

        # Calculate Correlation
        correlations = (
            df_analysis.corr()["error_magnitude"]
            .drop("error_magnitude")
            .sort_values(ascending=False)
        )

        print("Correlation between Error Magnitude and Input Features:")
        print(correlations)
    else:
        print("No inspiratory phase data found for analysis.")

    # 6. Submission Generation
    print("\n=== Submission Generation ===")
    SUBMISSION_THRESHOLD = 0.2164510190486908

    if val_mae < SUBMISSION_THRESHOLD:
        print(
            f"Validation Metric ({val_mae}) is better than threshold ({SUBMISSION_THRESHOLD})."
        )
        print("Generating submission file...")
        generate_predictions(config, load_cached_data=True)
    else:
        print(
            f"Validation Metric ({val_mae}) did not meet threshold ({SUBMISSION_THRESHOLD})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
