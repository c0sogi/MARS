import os
import torch
import numpy as np
import pandas as pd
from library.utils import seed_everything, LaplaceLogLikelihood
from library.data import get_dataloaders
from library.model import TSCPNet
from library.train import run_training
from library.predict import generate_submission


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    SEED = 42
    EPOCHS = 15  # Limited epochs for fast baseline
    BATCH_SIZE = 16
    THRESHOLD = -6.510164260864258
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure reproducibility
    seed_everything(SEED)

    print(
        f"Execution Context: Device={DEVICE}, Epochs={EPOCHS}, Batch Size={BATCH_SIZE}"
    )

    # ==========================================
    # 2. Model Training
    # ==========================================
    print("\n--- Starting Training ---")
    # run_training handles the training loop, early stopping, and saving the best model
    # We use debug=False to train on the full dataset, but limit epochs for speed.
    best_model_path = run_training(
        epochs=EPOCHS, batch_size=BATCH_SIZE, debug=False, seed=SEED
    )

    # ==========================================
    # 3. Validation & Metric Calculation
    # ==========================================
    print("\n--- Starting Validation Inference ---")

    # Load validation data
    # get_dataloaders returns (train_loader, val_loader, test_loader)
    _, val_loader, _ = get_dataloaders(
        batch_size=BATCH_SIZE, num_workers=2, load_cache=True, debug=False
    )

    # Load the best trained model
    model = TSCPNet().to(DEVICE)
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
        print(f"Loaded best model from {best_model_path}")
    else:
        print("Error: Best model not found. Using random weights.")

    model.eval()

    # Containers for predictions and targets
    all_targets = []
    all_fvc_preds = []
    all_sigma_preds = []

    # Inference loop (No gradients for speed/memory)
    with torch.no_grad():
        for batch in val_loader:
            # Move data to device
            img_ax = batch["img_ax"].to(DEVICE)
            img_cor = batch["img_cor"].to(DEVICE)
            tabular = batch["tabular"].to(DEVICE)
            target = batch["target"].to(DEVICE)
            weeks = batch["weeks"].to(DEVICE)
            base_fvc = batch["base_fvc"].to(DEVICE)

            # Forward pass
            alpha, sigma_base, sigma_growth = model(img_ax, img_cor, tabular)

            # Parametric Inference
            fvc_pred = base_fvc + alpha * weeks
            sigma_pred = sigma_base + sigma_growth * torch.abs(weeks)

            # Collect results
            all_targets.append(target.cpu().numpy())
            all_fvc_preds.append(fvc_pred.cpu().numpy())
            all_sigma_preds.append(sigma_pred.cpu().numpy())

    # Concatenate results
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_fvc_preds)
    sigma = np.concatenate(all_sigma_preds)

    # Compute Final Metric
    final_metric = LaplaceLogLikelihood(y_true, y_pred, sigma)
    print(f"Final Validation Metric: {final_metric}")

    # ==========================================
    # 4. Failure Analysis
    # ==========================================
    print("\n--- Failure Analysis ---")

    # Access the validation dataframe from the dataset
    val_df = val_loader.dataset.df.copy()

    # Ensure alignment
    if len(val_df) == len(y_true):
        # Calculate absolute error
        val_df["abs_error"] = np.abs(val_df["FVC"] - y_pred)

        # Features to correlate with error
        features = ["Weeks", "Percent", "Age", "Base_FVC"]

        print("Correlation between Absolute Error and Input Features:")
        for feat in features:
            if feat in val_df.columns:
                corr = val_df["abs_error"].corr(val_df[feat])
                print(f"  {feat}: {corr:.6f}")
            else:
                print(f"  {feat}: Not found in dataframe")
    else:
        print(
            f"Warning: Dataframe length ({len(val_df)}) mismatch with predictions ({len(y_true)}). Skipping detailed analysis."
        )

    # ==========================================
    # 5. Submission Generation
    # ==========================================
    print("\n--- Submission Check ---")
    if final_metric > THRESHOLD:
        print(
            f"Metric ({final_metric}) meets threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(
            model_path=best_model_path, batch_size=BATCH_SIZE, debug=False
        )
    else:
        print(
            f"Metric ({final_metric}) does not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
