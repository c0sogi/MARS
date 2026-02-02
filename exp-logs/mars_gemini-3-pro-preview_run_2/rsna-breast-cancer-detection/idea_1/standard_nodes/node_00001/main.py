import os
import torch
import pandas as pd
import numpy as np
import warnings

# Import from provided libraries
from library.config import (
    CACHE_DIR,
    DEVICE,
    SUBMISSION_DIR,
    seed_everything,
    BATCH_SIZE,
)
from library.train import run_training
from library.inference import predict_and_submit
from library.data import get_dataloaders, process_and_cache_metadata
from library.model import HybridEfficientNet
from library.utils import probabilistic_f1

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def analyze_failures(val_df, preds, targets):
    """
    Performs failure analysis by correlating prediction error with features.
    """
    print("\n=== Failure Analysis ===")

    # Calculate absolute error
    # Ensure preds and targets are numpy arrays
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    val_df = val_df.copy()
    val_df["prediction"] = preds
    val_df["target"] = targets
    val_df["error"] = np.abs(val_df["target"] - val_df["prediction"])

    # Identify feature columns (exclude IDs and targets)
    # We use the processed features which include OHE columns and normalized age
    exclude = [
        "patient_id",
        "image_id",
        "prediction_id",
        "file_path",
        "split",
        "cancer",
        "prediction",
        "target",
        "error",
    ]
    feature_cols = [c for c in val_df.columns if c not in exclude]

    # Calculate correlation
    correlations = {}
    for col in feature_cols:
        # Ensure column is numeric
        if pd.api.types.is_numeric_dtype(val_df[col]):
            corr = val_df[col].corr(val_df["error"])
            if not np.isnan(corr):
                correlations[col] = corr

    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Model Error:")
    for name, corr in sorted_corr[:5]:
        print(f"  {name}: {corr:.4f}")

    return sorted_corr


def main():
    # 1. Setup
    seed_everything()
    print("=== Starting Runfile Pipeline ===")

    # 2. Training
    # We use 2 epochs for a fast baseline as requested.
    # debug=False ensures we use the full dataset for a valid baseline,
    # fitting easily within the 2-hour limit on an A100.
    print("\n[Step 1/4] Training Model...")
    run_training(
        debug=False,
        load_cached_data=True,
        epochs=2,
        save_path=os.path.join(CACHE_DIR, "best_model.pth"),
    )

    # 3. Validation & Metrics
    print("\n[Step 2/4] Validating Model...")

    # Load processed metadata and loaders
    # We need the dataframe for failure analysis
    train_df, val_df, test_df, feature_cols = process_and_cache_metadata(
        load_cached_data=True
    )
    _, val_loader, _, num_tabular_features = get_dataloaders(
        load_cached_data=True, debug=False
    )

    # Load Model
    model = HybridEfficientNet(
        num_tabular_features=num_tabular_features,
        backbone_name="efficientnet_b0",
        pretrained=False,
    )
    model_path = os.path.join(CACHE_DIR, "best_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # Inference Loop on Validation
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for (images, tabular), targets in val_loader:
            images = images.to(DEVICE)
            tabular = tabular.to(DEVICE)

            logits = model((images, tabular))
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu())
            all_targets.append(targets.cpu())

    all_preds = torch.cat(all_preds).view(-1)
    all_targets = torch.cat(all_targets).view(-1)

    # Compute Metric
    pf1 = probabilistic_f1(all_targets, all_preds)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {pf1.item()}")

    # 4. Failure Analysis
    print("\n[Step 3/4] Performing Failure Analysis...")
    # Ensure val_df aligns with loader. Since shuffle=False and drop_last=False, they should align.
    # However, get_dataloaders might re-instantiate the dataset.
    # To be safe, we rely on the fact that both process_and_cache_metadata and get_dataloaders
    # read from the same cached parquet file in the same order.

    # Truncate val_df to match predictions if necessary (though they should match)
    if len(val_df) != len(all_preds):
        print(
            f"Warning: DataFrame length ({len(val_df)}) differs from predictions ({len(all_preds)}). Truncating to minimum."
        )
        min_len = min(len(val_df), len(all_preds))
        val_df = val_df.iloc[:min_len]
        all_preds = all_preds[:min_len]
        all_targets = all_targets[:min_len]

    analyze_failures(val_df, all_preds, all_targets)

    # 5. Submission
    print("\n[Step 4/4] Generating Submission...")
    predict_and_submit(
        model_path=model_path,
        output_path=os.path.join(SUBMISSION_DIR, "submission.csv"),
        debug=False,
        load_cached_data=True,
    )

    print("\n=== Pipeline Complete ===")


if __name__ == "__main__":
    main()
