import os
import sys
import torch
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, score_function
from library.data import get_dataloaders
from library.model import CIDSNet
from library.train import run_training


def predict_dataset(model, loader, device, processor, is_test=False):
    """
    Runs inference on a dataloader and returns a DataFrame with predictions.
    Handles inverse scaling of FVC and Sigma.
    """
    model.eval()
    preds_mu = []
    preds_sigma = []
    targets = []

    # Retrieve scaler stats for inverse transformation
    target_mean = processor.target_mean
    target_std = processor.target_std

    with torch.no_grad():
        for images, tabular, target_batch in loader:
            images = images.to(device)
            tabular = tabular.to(device)

            # Forward pass
            outputs = model(images, tabular)

            # Inverse Transform
            # Mu: z * std + mean
            mu_batch = outputs[:, 0].cpu().numpy() * target_std + target_mean
            # Sigma: z * std (Scale only, no mean shift)
            sigma_batch = outputs[:, 1].cpu().numpy() * target_std

            preds_mu.extend(mu_batch)
            preds_sigma.extend(sigma_batch)

            if not is_test:
                # Target: z * std + mean
                t_batch = target_batch.cpu().numpy() * target_std + target_mean
                targets.extend(t_batch)

    df = pd.DataFrame({"FVC_Pred": preds_mu, "Sigma_Pred": preds_sigma})

    if not is_test:
        df["FVC_True"] = targets

    return df


def main():
    # 1. Configuration & Setup
    seed_everything(Config.SEED)

    # Override Config for fast baseline execution
    Config.EPOCHS = 20  # Increased to 20 to allow full convergence (Cite solution_lesson_node_00100)
    Config.SUBMISSION_DIR = "./submission"  # Ensure output goes to the correct folder

    print(f"Configuration: EPOCHS={Config.EPOCHS}, DEVICE={Config.DEVICE}")

    # 2. Data Loading
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader, processor = get_dataloaders()

    # 3. Training
    print("Starting Training...")
    # run_training handles the loop, validation monitoring, and saving best_model.pth
    _ = run_training(epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE)

    # 4. Load Best Model for Analysis
    print("Loading best model for analysis...")
    device = torch.device(Config.DEVICE)
    model = CIDSNet().to(device)
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print("Warning: Checkpoint not found. Using last model state.")

    model.eval()

    # 5. Validation & Failure Analysis
    print("\nPerforming Validation Inference...")
    val_results = predict_dataset(model, val_loader, device, processor, is_test=False)

    # Calculate Final Metric
    final_metric = score_function(
        val_results["FVC_True"].values,
        val_results["FVC_Pred"].values,
        val_results["Sigma_Pred"].values,
    )

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation
    print("\n--- Failure Analysis ---")
    val_results["Abs_Error"] = np.abs(val_results["FVC_True"] - val_results["FVC_Pred"])

    # Retrieve original features from the dataset dataframe for correlation
    # The loader iterates sequentially (shuffle=False), so indices align
    df_val = val_loader.dataset.df.reset_index(drop=True)

    features_to_analyze = ["Age", "Weeks", "Baseline_FVC", "Percent"]
    print("Correlation between Absolute Error and Features:")

    for feat in features_to_analyze:
        if feat in df_val.columns:
            # Check for NaNs just in case
            valid_mask = ~df_val[feat].isna() & ~val_results["Abs_Error"].isna()
            if valid_mask.sum() > 1:
                corr, _ = pearsonr(
                    df_val.loc[valid_mask, feat],
                    val_results.loc[valid_mask, "Abs_Error"],
                )
                print(f"  {feat}: {corr:.4f}")
            else:
                print(f"  {feat}: Not enough data")
        else:
            print(f"  {feat}: Feature not found in validation dataframe")

    # 6. Submission Generation
    threshold = -6.573619738753321
    if final_metric > threshold:
        print(
            f"\nMetric ({final_metric}) > Threshold ({threshold}). Generating submission..."
        )

        # Predict on Test Set
        test_results = predict_dataset(
            model, test_loader, device, processor, is_test=True
        )

        # Get Patient_Week identifiers
        patient_weeks = test_loader.dataset.df["Patient_Week"].values

        # Construct Submission DataFrame
        sub_df = pd.DataFrame(
            {
                "Patient_Week": patient_weeks,
                "FVC": test_results["FVC_Pred"],
                "Confidence": test_results["Sigma_Pred"],
            }
        )

        # Apply Post-Processing Clip (Confidence >= 70)
        sub_df["Confidence"] = sub_df["Confidence"].apply(lambda x: max(x, 70))

        # Save
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        save_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        sub_df.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")

    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
