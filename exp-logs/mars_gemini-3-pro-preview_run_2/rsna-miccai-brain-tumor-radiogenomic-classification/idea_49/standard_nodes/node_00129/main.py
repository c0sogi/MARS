import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

# Ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config, set_seed
from library.utils import get_logger
from library.train import run_training
from library.inference import generate_submission
from library.model import AsymmetricEfficientNet
from library.data_loader import get_dataloaders


def pearson_corr(x, y):
    """Calculates Pearson correlation coefficient using NumPy."""
    x = np.array(x)
    y = np.array(y)
    if len(x) != len(y) or len(x) == 0:
        return 0.0
    mx = np.mean(x)
    my = np.mean(y)
    xm, ym = x - mx, y - my
    r_num = np.sum(xm * ym)
    r_den = np.sqrt(np.sum(xm * xm) * np.sum(ym * ym))
    if r_den == 0:
        return 0.0
    return r_num / r_den


def main():
    # 1. Setup
    set_seed(Config.SEED)
    logger = get_logger(name="Runfile")

    # 2. Training
    # Limiting to 5 epochs for a fast baseline execution
    logger.info("Starting training phase...")
    run_training(epochs=5, debug=False)

    # 3. Validation
    logger.info("Starting validation phase...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the best model saved during training
    model = AsymmetricEfficientNet()
    if not os.path.exists(Config.MODEL_PATH):
        logger.error("Best model not found. Exiting.")
        return

    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Load Validation Data
    # We read the CSV directly to ensure we have the metadata for analysis later
    val_df = pd.read_csv(Config.VAL_METADATA)
    loaders = get_dataloaders(val_df=val_df)
    val_loader = loaders["val"]

    all_targets = []
    all_preds = []

    # Inference loop
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()

            all_preds.extend(probs.flatten())
            all_targets.extend(targets.numpy().flatten())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Compute Metric
    try:
        val_auc = roc_auc_score(all_targets, all_preds)
    except Exception:
        val_auc = 0.5

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {val_auc}")

    # 4. Failure Analysis
    logger.info("Performing failure analysis...")
    errors = np.abs(all_targets - all_preds)

    # Extract 'FLAIR_slices' from filesystem as a feature for correlation
    # This checks if model performance degrades with scan depth
    slice_counts = []
    for _, row in val_df.iterrows():
        flair_path = os.path.join(Config.INPUT_DIR, row["path_FLAIR"])
        try:
            # Count files in the directory
            count = len([f for f in os.listdir(flair_path) if f.endswith(".dcm")])
        except Exception:
            count = 0
        slice_counts.append(count)

    corr_slices = pearson_corr(errors, slice_counts)
    print(f"Correlation between Error and FLAIR_slices: {corr_slices}")

    # 5. Submission
    threshold = 0.6321818181818182
    if val_auc > threshold:
        logger.info(
            f"Validation AUC ({val_auc}) > Threshold ({threshold}). Generating submission..."
        )
        generate_submission(load_cached_data=True)
    else:
        logger.info(
            f"Validation AUC ({val_auc}) <= Threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
