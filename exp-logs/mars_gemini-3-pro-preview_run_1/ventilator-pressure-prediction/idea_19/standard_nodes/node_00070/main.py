import sys
import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders
from library.model import VentilatorModel
from library.train import Trainer


def main():
    # 1. Setup
    # Initialize seed and directories
    seed_everything(Config.SEED)
    Config.setup()

    device = Config.DEVICE
    print(f"Starting execution on device: {device}")

    # 2. Data Loading
    # Load data using the library function, utilizing cache if available
    print("Initializing data loaders...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    print("Initializing model...")
    model = VentilatorModel()
    model.to(device)

    # 4. Training
    # Train the model using the provided Trainer class
    print("Starting training...")
    trainer = Trainer(model, train_loader, val_loader)
    trainer.fit()

    # 5. Load Best Model for Final Evaluation
    # The trainer saves the best model to Config.MODEL_PATH. We load it to ensure
    # we use the optimal weights for validation analysis and submission.
    print("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    # 6. Validation & Failure Analysis
    print("Running inference on validation set...")
    val_preds = []
    val_targets = []
    val_u_outs = []
    val_inputs_list = []

    # Optimized inference loop
    with torch.no_grad():
        for batch in val_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            u_out = batch["u_out"].to(device)

            # Forward pass
            final_out, _ = model(x)
            final_out = final_out.squeeze(-1)

            # Store results on CPU to save GPU memory
            val_preds.append(final_out.cpu().numpy())
            val_targets.append(y.cpu().numpy())
            val_u_outs.append(u_out.cpu().numpy())
            val_inputs_list.append(x.cpu().numpy())

    # Concatenate all batches
    val_preds = np.concatenate(val_preds)
    val_targets = np.concatenate(val_targets)
    val_u_outs = np.concatenate(val_u_outs)
    val_inputs = np.concatenate(val_inputs_list)  # Shape: (N, 80, F)

    # Flatten arrays for metric computation
    val_preds_flat = val_preds.flatten()
    val_targets_flat = val_targets.flatten()
    val_u_outs_flat = val_u_outs.flatten()

    # Filter for inspiratory phase (u_out == 0)
    insp_mask = val_u_outs_flat == 0

    # Compute MAE
    if np.sum(insp_mask) > 0:
        mae = np.mean(np.abs(val_preds_flat[insp_mask] - val_targets_flat[insp_mask]))
    else:
        mae = 0.0

    # Print the required validation metric
    print(f"Final Validation Metric: {mae}")

    # Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate absolute error
    errors = np.abs(val_preds_flat - val_targets_flat)
    errors_insp = errors[insp_mask]

    # Flatten inputs and filter by mask to align with errors
    val_inputs_flat = val_inputs.reshape(-1, val_inputs.shape[-1])
    val_inputs_insp = val_inputs_flat[insp_mask]

    # Feature names based on library/data.py structure
    feature_names = [
        "time_step",
        "u_in",
        "R",
        "C",
        "volume",
        "R_u_in",
        "vol_C",
        "u_in_diff1",
        "u_in_diff2",
    ]

    # Create DataFrame for correlation analysis
    # We only take the first len(feature_names) columns corresponding to the base features
    analysis_df = pd.DataFrame(
        val_inputs_insp[:, : len(feature_names)], columns=feature_names
    )
    analysis_df["error"] = errors_insp

    # Compute and print correlations
    correlations = analysis_df.corr()["error"].sort_values(ascending=False)
    print("Correlation between Error and Features (Inspiratory Phase):")
    print(correlations)

    # 7. Submission
    THRESHOLD = 0.2164510190486908
    if mae < THRESHOLD:
        print(
            f"\nValidation metric {mae} is better than threshold {THRESHOLD}. Generating submission..."
        )

        test_preds = []
        test_ids = []

        # Inference on Test Set
        with torch.no_grad():
            for batch in test_loader:
                x = batch["x"].to(device)
                ids = batch["ids"].to(device)

                final_out, _ = model(x)
                final_out = final_out.squeeze(-1)

                test_preds.append(final_out.cpu().numpy())
                test_ids.append(ids.cpu().numpy())

        # Flatten results
        test_preds = np.concatenate(test_preds).flatten()
        test_ids = np.concatenate(test_ids).flatten()

        # Create submission DataFrame
        submission_df = pd.DataFrame({"id": test_ids, "pressure": test_preds})

        # Ensure submission directory exists and save
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nValidation metric {mae} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
