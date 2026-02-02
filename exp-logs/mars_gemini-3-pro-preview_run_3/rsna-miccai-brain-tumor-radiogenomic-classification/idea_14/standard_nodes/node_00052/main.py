import os
import sys
import torch
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, get_device
from library.train import run_training
from library.predict import run_inference
from library.model import BraTS25DNet
from library.data_loader import get_val_loader


def perform_failure_analysis(val_auc):
    """
    Analyzes the validation results to find correlations between error and input features.
    """
    print("\n" + "=" * 40)
    print(" FAILURE ANALYSIS")
    print("=" * 40)

    device = get_device()

    # 1. Load Validation Data and Metadata
    # We need the dataframe to get meta-features (slice counts)
    val_df = pd.read_parquet(Config.VAL_META_PATH)

    # Get loader for predictions
    val_loader = get_val_loader(load_cached=True)

    # 2. Load Best Model
    model = BraTS25DNet()
    model.to(device)

    if os.path.exists(Config.MODEL_SAVE_PATH):
        state_dict = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print("Error: Model checkpoint not found for failure analysis.")
        return

    # 3. Run Inference on Validation Set
    model.eval()
    all_targets = []
    all_probs = []

    # Note: The loader returns (images, targets)
    # We assume the loader order matches the dataframe order because
    # get_val_loader calls prepare_data which reads the parquet file sequentially
    # and creates the dataset. The loader is created with shuffle=False.

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_targets.extend(targets.numpy())
            all_probs.extend(probs.flatten().tolist())

    # 4. Calculate Errors
    val_df["pred"] = all_probs
    val_df["target"] = (
        all_targets  # Ensure alignment, though val_df['MGMT_value'] should be same
    )
    val_df["error"] = np.abs(val_df["MGMT_value"] - val_df["pred"])

    # 5. Extract Meta-Features (Slice Counts)
    # The metadata contains lists of paths. We count them.
    modalities = ["flair", "t1w", "t1wce", "t2w"]
    for mod in modalities:
        col = f"{mod}_paths"
        # Handle potential None or empty lists
        val_df[f"{mod}_count"] = val_df[col].apply(
            lambda x: len(x) if x is not None else 0
        )

    val_df["total_slices"] = val_df[[f"{m}_count" for m in modalities]].sum(axis=1)

    # 6. Compute Correlations
    print("Correlation between Error Magnitude and Meta-features:")
    features_to_check = [f"{m}_count" for m in modalities] + ["total_slices"]

    for feat in features_to_check:
        if val_df[feat].std() > 0:  # Avoid constant input warning
            corr, _ = pearsonr(val_df["error"], val_df[feat])
            print(f" - {feat}: {corr:.4f}")
        else:
            print(f" - {feat}: N/A (Constant value)")


def main():
    # 1. Setup
    # Modify Config for Fast Baseline requirements
    Config.EPOCHS = 10
    seed_everything(Config.SEED)

    # 2. Run Training
    # This will train, validate, and save the best model to Config.MODEL_SAVE_PATH
    print("Starting Training Pipeline...")
    best_val_auc = run_training()

    # 3. Report Metric
    # STRICT FORMAT REQUIRED
    print(f"Final Validation Metric: {best_val_auc}")

    # 4. Failure Analysis
    perform_failure_analysis(best_val_auc)

    # 5. Submission
    threshold = 0.6978181818181817
    if best_val_auc > threshold:
        print(
            f"\nValidation metric ({best_val_auc}) > threshold ({threshold}). Generating submission..."
        )
        run_inference()
    else:
        print(
            f"\nValidation metric ({best_val_auc}) <= threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
