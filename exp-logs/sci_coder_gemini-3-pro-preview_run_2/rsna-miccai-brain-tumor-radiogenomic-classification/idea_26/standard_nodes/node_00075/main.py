import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

# Import provided library modules
from library.config import Config
from library.utils import set_seed
from library.data import get_dataloader
from library.model import AsymmetricEfficientNet
from library.train import run_training
from library.predict import predict_submission


def perform_failure_analysis(val_df, y_true, y_pred):
    """
    Analyzes model errors against metadata features (Slice Count).
    """
    print("\n=== Failure Analysis ===")

    # Calculate absolute error
    errors = np.abs(y_true - y_pred)

    # Extract FLAIR slice counts for validation subjects
    slice_counts = []
    for _, row in val_df.iterrows():
        flair_path = os.path.join(Config.INPUT_DIR, row["path_FLAIR"])
        try:
            # Fast count of files
            count = len([f for f in os.listdir(flair_path) if f.endswith(".dcm")])
        except Exception:
            count = 0
        slice_counts.append(count)

    slice_counts = np.array(slice_counts)

    # Calculate correlation
    if len(errors) > 1 and np.std(slice_counts) > 0:
        corr, _ = pearsonr(errors, slice_counts)
        print(f"Correlation between Error Magnitude and FLAIR Slice Count: {corr:.4f}")

        if abs(corr) > 0.1:
            direction = "increases" if corr > 0 else "decreases"
            print(f"Insight: Error tendency {direction} as volume depth increases.")
        else:
            print(
                "Insight: No significant linear correlation between error and volume depth."
            )
    else:
        print("Insufficient variance to calculate correlation.")


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    threshold = 0.6303636363636363

    print(f"Starting runfile.py execution on {device}...")

    # 2. Training
    # We use the default epochs from Config (15), which is fast enough for this dataset size.
    print("\n=== Starting Training Phase ===")
    run_training(num_epochs=Config.NUM_EPOCHS, load_cached_data=True)

    # 3. Final Validation Assessment
    print("\n=== Starting Final Validation Assessment ===")

    # Load validation data
    val_loader = get_dataloader("val", load_cached_data=True)

    # Initialize model and load best weights
    model = AsymmetricEfficientNet().to(device)
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        print("Error: Model checkpoint not found after training.")
        return

    checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    all_targets = []
    all_preds = []

    # Inference loop
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)

            # Forward pass
            logits = model(inputs)
            probs = torch.sigmoid(logits)

            all_preds.extend(probs.cpu().numpy().flatten())
            all_targets.extend(targets.numpy().flatten())

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)

    # Compute Metric
    try:
        final_metric = roc_auc_score(all_targets, all_preds)
    except ValueError:
        final_metric = 0.5

    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    # Reload metadata to map back to subjects
    val_df = pd.read_csv(Config.VAL_METADATA)
    perform_failure_analysis(val_df, all_targets, all_preds)

    # 5. Submission
    if final_metric > threshold:
        print(
            f"\nValidation metric ({final_metric}) > threshold ({threshold}). Generating submission..."
        )
        predict_submission(load_cached_data=True)
    else:
        print(
            f"\nValidation metric ({final_metric}) <= threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
