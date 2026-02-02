import os
import sys
import warnings
import torch
import pandas as pd
import numpy as np

# Import from provided libraries
from library.config import Config
from library.utils import seed_everything, laplace_log_likelihood_metric
from library.data import get_dataloaders
from library.model import AttentionFusedDualAxisNet
from library.train import Trainer, generate_submission

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def evaluate_validation(model, val_loader, device):
    """
    Runs inference on the validation set to compute the final metric
    and collect data for failure analysis.
    """
    model.eval()

    all_true = []
    all_pred = []
    all_sigma = []
    all_tabular = []
    all_weeks = []

    # Disable gradient calculation for inference efficiency
    with torch.no_grad():
        for batch in val_loader:
            # Move batch to device
            axial = batch["axial"].to(device)
            coronal = batch["coronal"].to(device)
            tabular = batch["tabular"].to(device)
            target = batch["target"].to(device)
            time_delta = batch["time_delta"].to(device)
            baseline_fvc = batch["baseline_fvc"].to(device)

            # Forward pass
            alpha, sigma_base, sigma_growth = model(axial, coronal, tabular)

            # Reconstruct FVC and Confidence from parametric outputs
            fvc_pred = baseline_fvc + alpha * time_delta
            sigma = sigma_base + sigma_growth * torch.abs(time_delta)

            # Accumulate results
            all_true.extend(target.cpu().numpy())
            all_pred.extend(fvc_pred.cpu().numpy())
            all_sigma.extend(sigma.cpu().numpy())
            all_tabular.extend(tabular.cpu().numpy())
            all_weeks.extend(time_delta.cpu().numpy())

    # Convert to numpy arrays
    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    sigma = np.array(all_sigma)
    tabular_data = np.array(all_tabular)
    weeks = np.array(all_weeks)

    # Compute Final Metric
    score = laplace_log_likelihood_metric(y_true, y_pred, sigma)

    # Construct DataFrame for Failure Analysis
    # Tabular features mapping: 0=Age(Norm), 1=Percent(Norm), 2+=OneHotEncodings
    analysis_df = pd.DataFrame(
        {
            "True_FVC": y_true,
            "Pred_FVC": y_pred,
            "Sigma": sigma,
            "Abs_Error": np.abs(y_true - y_pred),
            "Relative_Week": weeks,
            "Norm_Age": tabular_data[:, 0],
            "Norm_Percent": tabular_data[:, 1],
        }
    )

    return score, analysis_df


def analyze_failures(df):
    """
    Analyzes the relationship between model errors and input features.
    """
    print("Calculating correlations between Absolute Error and features...")

    # Select numerical features for correlation
    features = ["Norm_Age", "Norm_Percent", "Relative_Week", "Sigma"]

    # Compute correlation matrix
    correlations = df[["Abs_Error"] + features].corr()["Abs_Error"].drop("Abs_Error")

    print(correlations)

    # Additional stats
    mean_error = df["Abs_Error"].mean()
    print(f"Mean Absolute Error on Validation Set: {mean_error:.4f} ml")


def main():
    # 1. Setup
    seed_everything(Config.SEED)

    # Adjust Config for Fast Baseline
    # 20 epochs is sufficient for this dataset size and ensures <2h runtime
    Config.NUM_EPOCHS = 20

    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")

    # 2. Data Loading
    print("Loading datasets...")
    # debug=False ensures we use the full dataset for valid metrics
    train_loader, val_loader, test_loader = get_dataloaders(debug=Config.DEBUG)

    # 3. Model Initialization
    print("Initializing model...")
    model = AttentionFusedDualAxisNet().to(device)

    # 4. Training
    print("Starting training...")
    trainer = Trainer(model, train_loader, val_loader, device)
    trainer.fit()

    # 5. Validation Assessment
    print("\nLoading best model for validation assessment...")
    if not os.path.exists(trainer.best_model_path):
        print("Error: Best model checkpoint not found.")
        return

    # Load best weights
    model.load_state_dict(torch.load(trainer.best_model_path, map_location=device))

    # Evaluate
    val_score, val_analysis_df = evaluate_validation(model, val_loader, device)

    # REQUIRED OUTPUT: Print full precision metric
    print(f"Final Validation Metric: {val_score}")

    # 6. Failure Analysis
    print("\n--- Failure Analysis ---")
    analyze_failures(val_analysis_df)

    # 7. Conditional Submission
    submission_threshold = -6.510164260864258

    if val_score > submission_threshold:
        print(
            f"\nValidation metric ({val_score}) exceeds threshold ({submission_threshold})."
        )
        generate_submission(model, test_loader, device)
    else:
        print(
            f"\nValidation metric ({val_score}) is below threshold ({submission_threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
