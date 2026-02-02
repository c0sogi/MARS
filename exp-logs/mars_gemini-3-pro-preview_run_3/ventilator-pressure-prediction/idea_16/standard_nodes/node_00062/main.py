import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import seed_everything
from library.data import prepare_datasets
from library.model import PITHNet
from library.train import run_training


def main():
    # 1. Setup and Configuration
    seed_everything(Config.SEED)

    # Fast Baseline Configuration
    # We limit epochs to 10 to ensure the run completes well within the 2-hour limit.
    # We increase batch size to 256 to efficiently utilize the A100 GPU.
    EPOCHS = 10
    BATCH_SIZE = 256

    print(f"Starting Fast Baseline Run (Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE})...")

    # 2. Train Model
    # run_training handles data loading, feature engineering (with caching),
    # model initialization, and the training loop.
    run_training(epochs=EPOCHS, batch_size=BATCH_SIZE, load_cached_data=True)

    # 3. Validation & Failure Analysis
    print("\n=== Validation & Failure Analysis ===")

    # Load data loaders
    # Since run_training has already run, this will efficiently load processed data from cache.
    _, val_loader, test_loader, test_ids = prepare_datasets(
        load_cached_data=True, batch_size=BATCH_SIZE
    )

    device = torch.device(Config.DEVICE)
    model = PITHNet().to(device)

    # Load the best model saved during training
    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    print(f"Loading best model from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Run Inference on Validation Set
    val_preds = []
    val_targets = []
    val_u_out = []
    val_inputs = []

    print("Running inference on validation set...")
    with torch.no_grad():
        for batch in val_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            u_out = batch["u_out"].to(device)

            # Forward pass
            preds = model(x)

            # Move to CPU and store for analysis
            val_preds.append(preds.cpu().numpy())
            val_targets.append(y.cpu().numpy())
            val_u_out.append(u_out.cpu().numpy())
            val_inputs.append(x.cpu().numpy())

    # Flatten arrays for analysis
    val_preds = np.concatenate(val_preds).flatten()
    val_targets = np.concatenate(val_targets).flatten()
    val_u_out = np.concatenate(val_u_out).flatten()

    # Reshape inputs: (N_breaths, Seq_Len, N_feats) -> (N_total, N_feats)
    val_inputs = np.concatenate(val_inputs, axis=0)
    val_inputs = val_inputs.reshape(-1, val_inputs.shape[-1])

    # Calculate Metric (MAE on Inspiratory Phase)
    # The metric is only scored on the inspiratory phase (u_out == 0)
    insp_mask = val_u_out == 0

    abs_errors = np.abs(val_preds - val_targets)

    # Filter errors for inspiratory phase
    insp_errors = abs_errors[insp_mask]
    final_metric = np.mean(insp_errors)

    # Print the required metric string
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation between features and error magnitude
    print("\nFailure Analysis (Feature Correlations with Absolute Error):")
    insp_inputs = val_inputs[insp_mask]

    for i, feature_name in enumerate(Config.FEATURE_COLS):
        feat_values = insp_inputs[:, i]

        # Handle constant features (std=0) to avoid NaN in correlation
        if np.std(feat_values) < 1e-9:
            corr = 0.0
        else:
            # Calculate Pearson correlation
            corr = np.corrcoef(insp_errors, feat_values)[0, 1]

        print(f"{feature_name}: {corr:.4f}")

    # 4. Submission Generation
    THRESHOLD = 0.1642141044139862

    if final_metric < THRESHOLD:
        print(
            f"\nValidation metric {final_metric} < {THRESHOLD}. Generating submission..."
        )

        test_preds = []

        print("Running inference on test set...")
        with torch.no_grad():
            for batch in test_loader:
                x = batch["x"].to(device)
                # Test set has no targets
                preds = model(x)
                test_preds.append(preds.cpu().numpy())

        test_preds = np.concatenate(test_preds).flatten()
        test_ids_flat = test_ids.flatten()

        # Create Submission DataFrame
        sub_df = pd.DataFrame({"id": test_ids_flat, "pressure": test_preds})

        # Save to file
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(f"\nValidation metric {final_metric} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
