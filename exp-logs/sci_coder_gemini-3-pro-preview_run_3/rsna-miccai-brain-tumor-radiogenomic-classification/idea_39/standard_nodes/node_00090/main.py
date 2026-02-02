import sys
import os
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Ensure library can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, calculate_roc_auc
from library.train import run_training
from library.predict import generate_submission
from library.data_loader import get_dataloaders
from library.model import SSFNet


def main():
    # 1. Setup & Reproducibility
    seed_everything(Config.SEED)

    # 2. Training Phase
    # We use 10 epochs as a fast baseline to ensure execution within time limits
    # The run_training function handles data loading, caching, and model saving.
    print("Starting Training Phase...")
    best_auc = run_training(
        num_epochs=10, batch_size=Config.BATCH_SIZE, load_cached_data=True
    )

    # 3. Validation & Failure Analysis
    print("\nStarting Validation & Failure Analysis Phase...")
    device = torch.device(Config.DEVICE)

    # Load Validation Data
    # We use the loader to ensure we process data exactly as the model expects
    _, val_loader, _ = get_dataloaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=True
    )

    # Load Metadata for Analysis (to get slice counts for failure analysis)
    # The validation loader preserves the order of the parquet file (shuffle=False)
    if os.path.exists(Config.VAL_META_PATH):
        val_meta_df = pd.read_parquet(Config.VAL_META_PATH)
    else:
        print(f"Error: Validation metadata not found at {Config.VAL_META_PATH}")
        return

    # Load Best Model
    model = SSFNet()
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    else:
        print(
            "Warning: Best model not found, using initialized model (performance will be random)."
        )

    model.to(device)
    model.eval()

    # Inference on Validation Set
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for (even, odd), labels in val_loader:
            even = even.to(device)
            odd = odd.to(device)

            logits = model(even, odd)
            probs = torch.sigmoid(logits).view(-1).cpu().numpy()

            val_preds.extend(probs)
            val_targets.extend(labels.numpy())

    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)

    # Compute and Print Final Metric
    # Required format: "Final Validation Metric: <value>"
    final_metric = calculate_roc_auc(val_targets, val_preds)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlate Error with Slice Counts
    # Calculate absolute error
    errors = np.abs(val_targets - val_preds)

    # Ensure the length matches
    if len(val_meta_df) != len(errors):
        print(
            f"Warning: Metadata length ({len(val_meta_df)}) mismatches predictions ({len(errors)}). Skipping detailed analysis."
        )
    else:
        # Calculate slice counts per modality
        modalities = ["flair", "t1w", "t1wce", "t2w"]
        print("\nCorrelation between Absolute Error and Input Features (Slice Counts):")

        for mod in modalities:
            col_name = f"{mod}_paths"
            # Handle potential None or empty lists
            counts = val_meta_df[col_name].apply(
                lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0
            )

            # Compute correlation
            if np.std(counts) > 0 and np.std(errors) > 0:
                corr, _ = pearsonr(errors, counts)
                print(f"Error vs {mod} count: {corr:.4f}")
            else:
                print(f"Error vs {mod} count: NaN (Constant input)")

    # 4. Submission Generation
    # Threshold defined in task requirements
    THRESHOLD = 0.6978181818181817

    if final_metric > THRESHOLD:
        print(
            f"\nValidation metric ({final_metric}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(load_cached_data=True, batch_size=Config.BATCH_SIZE)
    else:
        print(
            f"\nValidation metric ({final_metric}) does not exceed threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
