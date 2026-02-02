import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

# Import from the provided library
from library.config import Config
from library.train import train, set_seed
from library.model import AsymmetricEfficientNet
from library.data import get_dataloaders
from library.inference import predict


def run_failure_analysis(val_df, val_preds, val_targets):
    """
    Performs failure analysis by correlating prediction errors with metadata features.
    """
    print("\n--- Failure Analysis ---")

    # Ensure lengths match
    if len(val_df) != len(val_preds):
        print(
            f"Warning: Metadata length ({len(val_df)}) matches predictions ({len(val_preds)})?"
        )
        # We assume the dataloader preserves order and the metadata file aligns with it.
        # The library data loader builds the list iterating through the dataframe rows sequentially.

    # Calculate Error
    val_df = val_df.copy()
    val_df["pred"] = val_preds
    val_df["target"] = val_targets
    val_df["error"] = np.abs(val_df["target"] - val_df["pred"])

    # Extract simple metadata features for correlation
    # We'll count files in the FLAIR directory as a proxy for scan depth/quality
    flair_counts = []
    for _, row in val_df.iterrows():
        try:
            path = os.path.join(Config.INPUT_DIR, row["path_FLAIR"])
            if os.path.exists(path):
                count = len([f for f in os.listdir(path) if f.endswith(".dcm")])
            else:
                count = 0
        except Exception:
            count = 0
        flair_counts.append(count)

    val_df["flair_slice_count"] = flair_counts

    # Calculate correlations
    # 1. Correlation with Slice Count
    if val_df["flair_slice_count"].std() > 0:
        corr_slices, _ = pearsonr(val_df["error"], val_df["flair_slice_count"])
        print(f"Correlation (Error vs FLAIR Slice Count): {corr_slices:.4f}")
    else:
        print("Correlation (Error vs FLAIR Slice Count): N/A (Constant values)")

    # 2. Correlation with Target (Class Bias)
    if val_df["target"].std() > 0:
        corr_target, _ = pearsonr(val_df["error"], val_df["target"])
        print(f"Correlation (Error vs Target Class): {corr_target:.4f}")
    else:
        print("Correlation (Error vs Target Class): N/A")


def main():
    # 1. Setup & Configuration
    # Set seed for reproducibility
    set_seed(Config.SEED)

    # Configure for a fast baseline run
    # We use 10 epochs which is enough for the small dataset (approx 500 samples)
    # but fast enough to complete quickly.
    FAST_EPOCHS = 10

    print("Starting Fast Baseline Run...")

    # 2. Train the Model
    # This handles data loading (with caching), training, and saving the best model.
    train(epochs=FAST_EPOCHS, batch_size=Config.BATCH_SIZE, load_cached_data=True)

    # 3. Validation Assessment
    print("\nPerforming Final Validation Assessment...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load DataLoaders again to get the validation set
    _, val_loader, _, _ = get_dataloaders(load_cached_data=True)

    # Load the best model
    model = AsymmetricEfficientNet()
    model.to(device)

    if os.path.exists(Config.BEST_MODEL_PATH):
        state_dict = torch.load(Config.BEST_MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print("Error: Best model checkpoint not found.")
        return

    # Inference on Validation Set
    model.eval()
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()

            val_preds.extend(probs)
            val_targets.extend(labels.numpy().flatten())

    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)

    # Calculate Metric
    if len(np.unique(val_targets)) > 1:
        final_metric = roc_auc_score(val_targets, val_preds)
    else:
        final_metric = 0.5

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    # Load validation metadata to correlate errors with features
    try:
        val_df = pd.read_csv(Config.VAL_METADATA_PATH)
        run_failure_analysis(val_df, val_preds, val_targets)
    except Exception as e:
        print(f"Skipping detailed failure analysis due to error: {e}")

    # 5. Conditional Submission
    THRESHOLD = 0.6254545454545455

    if final_metric > THRESHOLD:
        print(
            f"\nValidation metric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        predict(load_cached_data=True)
    else:
        print(
            f"\nValidation metric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
