import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

# Import from provided library files
from library.config import Config
from library.data_utils import get_dataloaders, set_seed, preprocess_data
from library.model_utils import IPPFEModel
from library.train_utils import train, generate_submission


def perform_failure_analysis(model, val_loader, df_val, device):
    """
    Analyzes model failure modes by correlating prediction errors with input features.
    """
    print("\nStarting Failure Analysis...")
    model.eval()
    all_preds = []
    all_targets = []

    # Generate predictions on validation set
    with torch.no_grad():
        for batch in val_loader:
            cat_x = batch["cat_features"].to(device)
            cont_x = batch["cont_features"].to(device)
            targets = batch["target"].to(device)

            logits = model(cat_x, cont_x)
            probs = torch.sigmoid(logits)

            # Average predictions across the 5 streams
            avg_probs = torch.mean(probs, dim=1)

            all_preds.extend(avg_probs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Calculate absolute error
    errors = np.abs(all_targets - all_preds)

    # Create a DataFrame for correlation analysis
    # We use the processed validation dataframe which aligns with the loader
    # Ensure we only correlate with numeric columns
    analysis_df = df_val.select_dtypes(include=[np.number]).copy()

    # Remove target and id if present to avoid spurious correlations
    cols_to_drop = [c for c in ["target", "id"] if c in analysis_df.columns]
    analysis_df.drop(columns=cols_to_drop, inplace=True, errors="ignore")

    # Calculate correlations
    correlations = analysis_df.corrwith(pd.Series(errors, index=analysis_df.index))

    # Sort by absolute correlation
    abs_corrs = correlations.abs().sort_values(ascending=False)

    print("Top 5 Features correlated with Error Magnitude:")
    for feat in abs_corrs.head(5).index:
        corr_val = correlations[feat]
        print(f"{feat}: {corr_val:.6f}")


def main():
    # 1. Setup and Config Overrides
    # Override Config for fast baseline execution while maintaining enough capacity to hit the score
    Config.EPOCHS = 15  # Reduced from 50 to meet time constraints
    Config.MAX_SAMPLES = None  # Use full dataset to ensure high AUC

    # Ensure submission directory exists and update path
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)
    Config.SUBMISSION_PATH = os.path.join(submission_dir, "submission.csv")

    # Set seeds for reproducibility
    set_seed(Config.SEED)

    device = torch.device(Config.DEVICE)

    # 2. Data Loading
    print("Loading and preprocessing data...")
    # We need the raw dataframes for failure analysis later
    df_train, df_val, df_test, metadata = preprocess_data(load_cached_data=True)

    # Get dataloaders
    train_loader, val_loader, test_loader, _ = get_dataloaders(load_cached_data=True)

    # 3. Training
    print(f"Training model on {device}...")
    best_auc = train(train_loader, val_loader, metadata)

    # 4. Final Validation Metric
    # The requirement asks to print the full precision metric
    print(f"Final Validation Metric: {best_auc}")

    # 5. Failure Analysis
    # Load the best model for analysis
    model = IPPFEModel(
        vocab_sizes=metadata["vocab_sizes"], num_cont=metadata["num_cont_features"]
    ).to(device)

    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    else:
        print("Warning: Best model file not found. Using current model state.")

    perform_failure_analysis(model, val_loader, df_val, device)

    # 6. Conditional Submission
    THRESHOLD = 0.9975746465492954

    if best_auc > THRESHOLD:
        print(
            f"\nValidation AUC ({best_auc}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(test_loader, metadata)
    else:
        print(
            f"\nValidation AUC ({best_auc}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
