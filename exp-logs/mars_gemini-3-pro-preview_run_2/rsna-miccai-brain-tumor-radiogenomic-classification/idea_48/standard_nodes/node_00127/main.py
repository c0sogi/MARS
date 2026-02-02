import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

# Import provided library modules
from library.config import WORKING_DIR, METADATA_DIR, INPUT_DIR, DEVICE, BATCH_SIZE
from library.trainer import run_training, predict_and_submit
from library.data_loader import get_dataloader
from library.model_factory import AsymmetricEfficientNet
from library.utils import set_seed


def perform_failure_analysis(df_val, all_targets, all_preds):
    """
    Analyzes the correlation between prediction error and input features (slice count).
    """
    print("Performing Failure Analysis...")

    # Calculate absolute error
    errors = np.abs(all_targets - all_preds)

    # Extract feature: Number of slices in FLAIR modality
    # This serves as a proxy for scan resolution/volume size
    flair_counts = []
    for _, row in df_val.iterrows():
        flair_path = os.path.join(INPUT_DIR, row["path_FLAIR"])
        if os.path.exists(flair_path):
            # Count files in directory
            try:
                count = len([f for f in os.listdir(flair_path) if f.endswith(".dcm")])
            except Exception:
                count = 0
        else:
            count = 0
        flair_counts.append(count)

    flair_counts = np.array(flair_counts)

    # Compute correlation
    if len(errors) > 1 and np.std(errors) > 0 and np.std(flair_counts) > 0:
        corr = np.corrcoef(errors, flair_counts)[0, 1]
        print(f"Correlation between Error and FLAIR Slice Count: {corr:.6f}")
    else:
        print("Could not calculate correlation (insufficient variance or data).")


def main():
    # 1. Setup
    set_seed()
    print("Orchestration script started.")

    # 2. Training
    # run_training handles loading data, training loop, and saving the best model
    best_model_path = run_training()

    # 3. Validation
    print("Loading best model for final validation...")

    # Load validation metadata and loader
    val_csv_path = os.path.join(METADATA_DIR, "val.csv")
    df_val = pd.read_csv(val_csv_path)
    val_loader = get_dataloader(
        df_val, BATCH_SIZE, phase="val", load_cached_anchors=True
    )

    # Load Model
    model = AsymmetricEfficientNet().to(DEVICE)
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    model.eval()

    # Inference
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(DEVICE)
            # Forward pass
            outputs = model(inputs)
            preds = torch.sigmoid(outputs).cpu().numpy().flatten()

            all_preds.extend(preds)
            all_targets.extend(targets.numpy().flatten())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Calculate Metric
    try:
        final_metric = roc_auc_score(all_targets, all_preds)
    except ValueError:
        final_metric = 0.5

    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    perform_failure_analysis(df_val, all_targets, all_preds)

    # 5. Submission
    THRESHOLD = 0.6321818181818182
    if final_metric > THRESHOLD:
        print(
            f"Metric ({final_metric}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        predict_and_submit(best_model_path)
    else:
        print(
            f"Metric ({final_metric}) does not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
