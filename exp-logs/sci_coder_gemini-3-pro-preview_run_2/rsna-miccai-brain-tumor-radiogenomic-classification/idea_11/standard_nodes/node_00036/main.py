import os
import sys
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

# Import from provided libraries
from library.config import Config
from library.data import get_dataloader, set_seed
from library.model import AsymmetricEfficientNet
from library.train import run_training, predict_and_submit


def perform_failure_analysis(model, val_loader, device):
    """
    Analyzes model failure modes on the validation set.
    Calculates correlation between error magnitude and metadata features.
    """
    print("\n--- Performing Failure Analysis ---")

    model.eval()
    all_preds = []
    all_labels = []
    all_ids = []

    # 1. Get Predictions
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)

            # Forward pass
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            all_preds.extend(probs)
            all_labels.extend(labels.numpy().flatten())

            # Note: The loader shuffles if shuffle=True.
            # library.data.get_dataloader sets shuffle=False for 'val' split implicitly
            # (shuffle is only True for 'train').
            # However, to be safe and map back to metadata, we rely on the order
            # preserved by the sequential sampler in validation.

    # 2. Load Validation Metadata to extract features
    df_val = pd.read_csv(Config.VAL_METADATA)

    # Ensure alignment
    if len(df_val) != len(all_preds):
        print("Warning: Mismatch between validation set size and predictions.")
        return

    # Calculate Error Magnitude
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    errors = np.abs(y_true - y_pred)

    df_val["error"] = errors
    df_val["prediction"] = y_pred

    # 3. Extract Metadata Features for Correlation
    # We will count slices in each modality folder to see if volume depth affects error
    print("Extracting metadata features for correlation analysis...")

    features = {}
    modalities = ["FLAIR", "T1w", "T1wCE", "T2w"]

    # Pre-initialize lists
    for mod in modalities:
        features[f"{mod}_count"] = []

    for idx, row in df_val.iterrows():
        for mod in modalities:
            path_rel = row[f"path_{mod}"]
            full_path = os.path.join(Config.INPUT_DIR, path_rel)
            try:
                # Fast count of files
                count = len(
                    [
                        name
                        for name in os.listdir(full_path)
                        if os.path.isfile(os.path.join(full_path, name))
                    ]
                )
            except Exception:
                count = 0
            features[f"{mod}_count"].append(count)

    # Add features to DataFrame
    for key, values in features.items():
        df_val[key] = values

    # 4. Calculate Correlations
    print("\nCorrelation between Error Magnitude and Input Features:")
    numeric_cols = [c for c in df_val.columns if "count" in c or c == "MGMT_value"]

    for col in numeric_cols:
        if df_val[col].std() > 0:  # Avoid constant columns
            corr, _ = pearsonr(df_val["error"], df_val[col])
            print(f"Feature: {col:15s} | Correlation: {corr:.4f}")
        else:
            print(f"Feature: {col:15s} | Correlation: NaN (Constant)")


def main():
    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    # Override Config for Fast Baseline
    # Increased to 20 to allow convergence with higher dropout
    Config.NUM_EPOCHS = 20

    # Ensure reproducibility
    set_seed(Config.SEED)

    print(f"Starting Runfile Execution on {Config.DEVICE}")
    print(f"Working Directory: {Config.WORKING_DIR}")

    # --------------------------------------------------------------------------
    # 2. Training
    # --------------------------------------------------------------------------
    # run_training handles the training loop, saving the best model to Config.CHECKPOINT_PATH
    run_training()

    # --------------------------------------------------------------------------
    # 3. Validation & Metrics
    # --------------------------------------------------------------------------
    print("\n--- Final Validation Evaluation ---")

    # Load the best model saved during training
    model = AsymmetricEfficientNet().to(Config.DEVICE)
    if os.path.exists(Config.CHECKPOINT_PATH):
        model.load_state_dict(
            torch.load(Config.CHECKPOINT_PATH, map_location=Config.DEVICE)
        )
        print(f"Loaded best model weights from {Config.CHECKPOINT_PATH}")
    else:
        print("Error: Checkpoint not found. Cannot evaluate.")
        return

    model.eval()

    # Get validation loader
    val_loader = get_dataloader("val", load_cached_data=True)

    # Compute Metric on full validation set
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(Config.DEVICE)

            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_preds.extend(probs)
            all_labels.extend(labels.numpy())

    final_auc = roc_auc_score(all_labels, all_preds)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_auc}")

    # --------------------------------------------------------------------------
    # 4. Failure Analysis
    # --------------------------------------------------------------------------
    perform_failure_analysis(model, val_loader, Config.DEVICE)

    # --------------------------------------------------------------------------
    # 5. Submission
    # --------------------------------------------------------------------------
    # Threshold defined in task description
    THRESHOLD = 0.6254545454545455

    if final_auc > THRESHOLD:
        print(
            f"\nValidation metric ({final_auc}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        predict_and_submit()
    else:
        print(
            f"\nValidation metric ({final_auc}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
