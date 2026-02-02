import sys
import os
import pandas as pd
import numpy as np
import torch
from scipy.stats import pearsonr

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, get_device
from library.data import get_dataloaders
from library.train import Trainer


def perform_failure_analysis(trainer, val_loader):
    """
    Analyzes the correlation between model error and input data characteristics
    on the validation set.
    """
    print("\n" + "=" * 40)
    print(" FAILURE ANALYSIS")
    print("=" * 40)

    # 1. Load Validation Metadata to get features (slice counts, etc.)
    if not os.path.exists(Config.VAL_METADATA_PATH):
        print("Validation metadata not found. Skipping detailed analysis.")
        return

    val_df = pd.read_parquet(Config.VAL_METADATA_PATH)

    # 2. Get Model Predictions on Validation Set
    # Note: val_loader is not shuffled, so order matches val_df
    device = get_device()
    trainer.model.eval()

    # Load best weights for analysis
    if os.path.exists(trainer.best_model_path):
        trainer.model.load_state_dict(
            torch.load(trainer.best_model_path, map_location=device)
        )

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = trainer.model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()

            all_preds.extend(probs)
            all_targets.extend(targets.numpy().flatten())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # 3. Calculate Error
    errors = np.abs(all_targets - all_preds)

    # 4. Extract Meta-Features and Correlate
    # We calculate the number of slices available in the source metadata for each modality
    # to see if data quantity correlates with error.

    modalities = ["flair", "t1w", "t1wce", "t2w"]
    correlations = {}

    print(f"Analyzing {len(val_df)} validation samples...")

    try:
        # Ensure lengths match
        if len(val_df) != len(errors):
            print(
                f"Warning: Metadata length ({len(val_df)}) != Prediction length ({len(errors)}). Skipping correlation."
            )
            return

        for mod in modalities:
            col_name = f"{mod}_paths"
            # Calculate count of files for this modality
            counts = val_df[col_name].apply(lambda x: len(x) if x is not None else 0)

            # Calculate correlation with error
            if len(counts) > 1 and np.std(counts) > 0:
                corr, _ = pearsonr(counts, errors)
                correlations[f"{mod}_slice_count"] = corr
            else:
                correlations[f"{mod}_slice_count"] = 0.0

        # Also correlate with total slices
        total_slices = sum(
            val_df[f"{m}_paths"].apply(lambda x: len(x) if x is not None else 0)
            for m in modalities
        )
        corr_total, _ = pearsonr(total_slices, errors)
        correlations["total_slice_count"] = corr_total

        print("\nCorrelation between Error Magnitude and Input Features:")
        for feature, corr in correlations.items():
            print(f"  {feature}: {corr:.4f}")

    except Exception as e:
        print(f"Error during failure analysis: {e}")


def main():
    # 1. Setup
    seed_everything(Config.SEED)

    # 2. Data Loading
    # Use cached data if available to speed up execution
    print("Initializing Data Loaders...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Training
    print("Initializing Trainer...")
    trainer = Trainer(train_loader, val_loader, test_loader)

    print("Starting Training...")
    # Training for the configured number of epochs (15)
    trainer.fit(epochs=Config.EPOCHS, patience=5)

    # 4. Validation Metric
    # The trainer tracks the best AUC achieved on the validation set
    final_metric = trainer.best_auc
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    perform_failure_analysis(trainer, val_loader)

    # 6. Submission
    # Threshold check as per requirements
    THRESHOLD = 0.6978181818181817

    if final_metric > THRESHOLD:
        print(
            f"\nValidation metric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        trainer.generate_submission()
    else:
        print(
            f"\nValidation metric ({final_metric}) <= Threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
