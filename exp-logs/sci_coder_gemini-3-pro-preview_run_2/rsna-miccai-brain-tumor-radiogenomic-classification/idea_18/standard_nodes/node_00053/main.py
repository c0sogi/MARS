import os
import sys
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

# Import provided library modules
from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders
from library.model import ModalityAwareEfficientNet
from library.train import run_training
from library.predict import generate_submission


def main():
    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    seed_everything(Config.SEED)
    Config.EPOCHS = 10
    device = torch.device(Config.DEVICE)

    # --------------------------------------------------------------------------
    # 2. Training
    # --------------------------------------------------------------------------
    run_training(load_cached_data=True)

    # --------------------------------------------------------------------------
    # 3. Validation & Failure Analysis
    # --------------------------------------------------------------------------
    _, val_loader, _ = get_dataloaders(load_cached_data=True)

    model = ModalityAwareEfficientNet()
    model.to(device)

    if os.path.exists(Config.MODEL_PATH):
        state_dict = torch.load(Config.MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print("Error: Model checkpoint not found.")
        return

    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)

            logits = model(inputs)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_preds.extend(probs)
            all_targets.extend(targets.numpy())

    all_preds = np.array(all_preds).flatten()
    all_targets = np.array(all_targets).flatten()

    # Calculate Final Validation Metric (ROC AUC)
    # We handle the edge case where the validation set might only contain one class
    if len(np.unique(all_targets)) < 2:
        val_auc = 0.5
    else:
        val_auc = roc_auc_score(all_targets, all_preds)

    # Print the metric with full precision as required
    print(f"Final Validation Metric: {val_auc}")

    # --- Failure Analysis ---
    # Calculate absolute error for each sample
    errors = np.abs(all_targets - all_preds)

    # Load metadata to extract features for correlation analysis
    # The val_loader is non-shuffled and derived sequentially from Config.VAL_METADATA,
    # so the order of rows in df_val matches the order of predictions.
    df_val = pd.read_csv(Config.VAL_METADATA)

    # We correlate the model's error with the number of slices in each modality.
    # This helps identify if the model struggles with patients having limited data (fewer slices).
    modalities = ["FLAIR", "T1w", "T1wCE", "T2w"]

    print("Failure Analysis - Correlation of Error with Input Features:")

    for mod in modalities:
        # Extract slice counts for this modality
        counts = []
        for rel_path in df_val[f"path_{mod}"]:
            full_path = os.path.join(Config.INPUT_DIR, rel_path)
            if os.path.exists(full_path):
                # Count .dcm files in the directory
                try:
                    num_files = len(
                        [f for f in os.listdir(full_path) if f.endswith(".dcm")]
                    )
                except:
                    num_files = 0
                counts.append(num_files)
            else:
                counts.append(0)

        # Calculate Pearson correlation
        if len(counts) == len(errors):
            # Avoid warning if input is constant
            if np.std(counts) > 0 and np.std(errors) > 0:
                corr, _ = pearsonr(counts, errors)
            else:
                corr = 0.0
            print(f"Correlation with {mod} slice count: {corr}")
        else:
            print(f"Skipping {mod}: Length mismatch between metadata and predictions.")

    # --------------------------------------------------------------------------
    # 4. Submission
    # --------------------------------------------------------------------------
    # The threshold is specified in the task requirements.
    THRESHOLD = 0.6254545454545455

    if val_auc > THRESHOLD:
        # Generate predictions for the test set and save submission.csv
        generate_submission(load_cached_data=True)
    else:
        print(
            f"Validation metric ({val_auc}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
