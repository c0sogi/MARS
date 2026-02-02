import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# Import provided library modules
from library.config import Config
from library.dataset import SSVEDataset
from library.model import SSVEModel
from library.train import run_training, set_seed
from library.predict import run_inference


def perform_failure_analysis(val_ids, targets, preds):
    """
    Analyzes the correlation between model error and input meta-features.
    """
    print("\n" + "=" * 40)
    print(" FAILURE ANALYSIS")
    print("=" * 40)

    # Load metadata to get features
    try:
        df_val = pd.read_parquet(Config.VAL_META_PATH)
        # Ensure alignment by BraTS21ID
        df_val["BraTS21ID"] = df_val["BraTS21ID"].astype(str)

        # Create a dataframe for analysis
        analysis_df = pd.DataFrame(
            {"BraTS21ID": val_ids, "target": targets, "pred": preds}
        )

        # Calculate Error Magnitude
        analysis_df["error"] = np.abs(analysis_df["target"] - analysis_df["pred"])

        # Merge with metadata
        merged_df = pd.merge(analysis_df, df_val, on="BraTS21ID", how="left")

        # Extract Meta-Features (Slice Counts)
        modalities = ["flair", "t1w", "t1wce", "t2w"]
        correlations = {}

        print("Correlations between Error Magnitude and Slice Counts:")
        for mod in modalities:
            col_name = f"{mod}_paths"
            # Calculate slice count
            merged_df[f"{mod}_count"] = merged_df[col_name].apply(
                lambda x: len(x) if x is not None else 0
            )

            # Compute correlation
            if merged_df[f"{mod}_count"].std() > 0:
                corr = merged_df["error"].corr(merged_df[f"{mod}_count"])
                correlations[f"{mod}_count"] = corr
                print(f" - {mod}_count: {corr:.4f}")
            else:
                print(f" - {mod}_count: NaN (No variance)")

    except Exception as e:
        print(f"Failure analysis could not be completed: {e}")


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Training
    # run_training handles the training loop and saves the best model to Config.MODEL_PATH
    print("Starting Training Pipeline...")
    run_training(load_cached_data=True)

    # 3. Validation & Metric Calculation
    print("\nStarting Validation Evaluation...")

    # Load Validation Data
    val_dataset = SSVEDataset(mode="val", load_cached_data=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load Best Model
    model = SSVEModel()
    if not os.path.exists(Config.MODEL_PATH):
        print("Error: Model file not found. Training may have failed.")
        sys.exit(1)

    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    all_targets = []
    all_preds = []

    # Inference Loop (Multi-View Ensemble)
    with torch.no_grad():
        for images, targets in val_loader:
            # images shape: (Batch, 2, 64, 256, 256)
            targets_np = targets.numpy()

            # Move to device
            view_a = images[:, 0, ...].to(device)
            view_b = images[:, 1, ...].to(device)

            # Forward passes
            logits_a = model(view_a)
            logits_b = model(view_b)

            # Probabilities
            probs_a = torch.sigmoid(logits_a)
            probs_b = torch.sigmoid(logits_b)

            # Ensemble Average
            avg_probs = (probs_a + probs_b) / 2.0

            all_preds.extend(avg_probs.cpu().numpy().flatten())
            all_targets.extend(targets_np)

    # Calculate Metric
    val_auc = roc_auc_score(all_targets, all_preds)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_auc}")

    # 4. Failure Analysis
    perform_failure_analysis(val_dataset.get_ids(), all_targets, all_preds)

    # 5. Submission
    THRESHOLD = 0.6978181818181817

    if val_auc > THRESHOLD:
        print(
            f"\nValidation AUC ({val_auc}) > Threshold ({THRESHOLD}). Generating Submission..."
        )
        run_inference(load_cached_data=True)
    else:
        print(
            f"\nValidation AUC ({val_auc}) <= Threshold ({THRESHOLD}). Skipping Submission."
        )


if __name__ == "__main__":
    main()
