import os
import torch
import numpy as np
import pandas as pd

# Import from provided libraries
from library.config import Config
from library.train import run_training, set_seed
from library.dataset import get_dataloaders, LungDataset, get_transforms
from library.model import H2DAN
from library.loss import LaplaceLogLikelihoodLoss


def main():
    # 1. Configure for Fast Baseline
    # We override config parameters to ensure execution within time limits
    # while maintaining enough capacity to learn useful features.
    Config.EPOCHS = 15
    Config.BATCH_SIZE = 32  # Increase batch size for A100 efficiency
    Config.NUM_WORKERS = 4

    # Ensure reproducibility
    set_seed(Config.SEED)

    print("Starting Fast Baseline Run...")

    # 2. Run Training
    # This handles the training loop, validation, and saving the best model.
    # run_training returns the path to the best checkpoint.
    best_model_path = run_training()

    # 3. Load Resources for Analysis and Inference
    device = torch.device(Config.DEVICE)

    # We need to recreate the dataloaders to get access to the scaler and validation set metadata.
    # Since seeds are fixed, the scaler fit and data splits are deterministic.
    train_loader, val_loader, scaler = get_dataloaders(batch_size=Config.BATCH_SIZE)

    # Determine tabular input dimension from a sample batch
    sample_batch = next(iter(train_loader))
    tabular_dim = sample_batch["deep_tab"].shape[1]

    # Initialize and load the best model
    model = H2DAN(tabular_input_dim=tabular_dim)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    model.eval()

    # 4. Validation & Failure Analysis
    print("\nRunning Validation Analysis...")

    val_preds_fvc = []
    val_preds_sigma = []
    val_targets = []

    criterion = LaplaceLogLikelihoodLoss().to(device)
    total_metric = 0.0
    count = 0

    with torch.no_grad():
        for batch in val_loader:
            # Move to device
            batch_device = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            outputs = model(batch_device)

            pred_fvc = outputs["fvc_pred"]
            pred_sigma = outputs["confidence_pred"]
            target = batch_device["target"]

            # Calculate batch metric
            # The criterion returns the Loss (negative metric), so we negate it back.
            loss = criterion(pred_fvc, pred_sigma, target)
            total_metric += (-loss.item()) * target.size(0)  # Weighted by batch size
            count += target.size(0)

            # Store for analysis
            val_preds_fvc.extend(pred_fvc.cpu().numpy())
            val_preds_sigma.extend(pred_sigma.cpu().numpy())
            val_targets.extend(target.cpu().numpy())

    final_val_metric = total_metric / count
    print(f"Final Validation Metric: {final_val_metric}")

    # Failure Analysis
    # Access the underlying dataframe from the validation dataset
    # Note: val_loader has shuffle=False, so order is preserved.
    val_df = val_loader.dataset.df.copy()

    # Verify alignment
    if len(val_df) != len(val_preds_fvc):
        print(
            f"Warning: Validation DF length ({len(val_df)}) != Predictions length ({len(val_preds_fvc)})"
        )
        min_len = min(len(val_df), len(val_preds_fvc))
        val_df = val_df.iloc[:min_len]
        val_preds_fvc = val_preds_fvc[:min_len]
        val_targets = val_targets[:min_len]

    val_df["Pred_FVC"] = val_preds_fvc
    val_df["True_FVC"] = val_targets
    val_df["Abs_Error"] = np.abs(val_df["True_FVC"] - val_df["Pred_FVC"])

    print("\nFailure Analysis (Correlation with Absolute Error):")
    features_to_check = [
        "Age",
        "Percent",
        "Weeks",
        "Baseline_FVC",
        "Baseline_Percent",
        "Delta_Week",
    ]

    # Filter for features that exist in the dataframe
    features_present = [f for f in features_to_check if f in val_df.columns]

    # Calculate correlations
    if features_present:
        correlations = (
            val_df[features_present + ["Abs_Error"]]
            .corr()["Abs_Error"]
            .drop("Abs_Error")
        )
        print(correlations)
    else:
        print("No relevant features found for correlation analysis.")

    # 5. Submission Generation
    threshold = -6.510164260864258

    if final_val_metric > threshold:
        print(
            f"\nMetric ({final_val_metric}) > Threshold ({threshold}). Generating submission..."
        )

        # Prepare Test Loader using the fitted scaler
        test_ds = LungDataset(
            mode="test", transform=get_transforms("val"), scaler=scaler
        )
        test_loader = torch.utils.data.DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        submission_rows = []

        with torch.no_grad():
            for batch in test_loader:
                batch_device = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                outputs = model(batch_device)

                pred_fvc = outputs["fvc_pred"].cpu().numpy()
                pred_sigma = outputs["confidence_pred"].cpu().numpy()
                patient_week_ids = batch["patient_week_id"]

                for pw, fvc, sigma in zip(patient_week_ids, pred_fvc, pred_sigma):
                    # Clip confidence at 70 ml as per task description constraints
                    sigma_clipped = max(sigma, 70)

                    submission_rows.append(
                        {"Patient_Week": pw, "FVC": fvc, "Confidence": sigma_clipped}
                    )

        submission_df = pd.DataFrame(submission_rows)

        # Ensure output directory exists and save
        os.makedirs(os.path.dirname(Config.SUBMISSION_FILE), exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")
        print(submission_df.head())

    else:
        print(
            f"\nMetric ({final_val_metric}) <= Threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
